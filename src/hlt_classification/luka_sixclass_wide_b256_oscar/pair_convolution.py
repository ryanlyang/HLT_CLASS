"""Avoid the large fallback's col2im input-gradient path for pointwise pairs."""
import torch
from torch import nn
from torch.autograd.function import once_differentiable

INDEX_LIMIT = 2**31-1
POLICY = dict(version='sixclass_pair_pointwise_dgrad_v1', scope='CONCAT_native_pair_conv1d',
    trigger='input_or_output_elements_per_sample_exceed_int32', trigger_elements=INDEX_LIMIT,
    forward='native_conv1d', parameter_gradients='native_convolution_backward',
    input_gradient='weight_transpose_matmul_grad_output', sparse_batch_dimension=1,
    batchnorm='unchanged_full_population', dtype='native_autocast',
    modes=['training_grad_enabled'], inference='native_unchanged',
    clipping=False, truncation=False, microbatching=False)


class _Pointwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,stats):
        ctx.save_for_backward(x,weight)
        ctx.bias_shape=None if bias is None else list(bias.shape)
        ctx.stats=stats
        return torch.nn.functional.conv1d(x,weight,bias)

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        x,weight=ctx.saved_tensors
        dy=dy.contiguous()
        need_x,need_w,need_b,_=ctx.needs_input_grad
        need_b=need_b and ctx.bias_shape is not None
        dw=db=None
        with torch.autocast(dy.device.type,enabled=False):
            if need_w or need_b:
                _,dw,db=torch.ops.aten.convolution_backward.default(
                    dy,x,weight,ctx.bias_shape,[1],[0],[1],False,[0],1,[False,need_w,need_b])
            # Release the restored input before allocating its large gradient.
            del x
            dx=None
            if need_x:
                # No spatial folding is needed for a 1x1 convolution. This is
                # the native fallback's GEMM without the spatial col2im step.
                dx=torch.mm(weight.squeeze(-1).T,dy.squeeze(0)).unsqueeze(0)
                ctx.stats['input_gradient_calls']+=1
        return dx,dw,db,None


class PointwiseInputGradient:
    """Instance-local native checkpoint keys; no replacement of BN or activations."""
    def __init__(self,pair):
        self.force=False  # Parity alone forces this path on small arrays.
        self.stats=dict(calls=0,fallback_calls=0,input_gradient_calls=0,largest_pair_count=0)
        self.layers=[]
        for name,layer in pair.named_modules():
            if not isinstance(layer,nn.Conv1d):continue
            if (layer.kernel_size!=(1,) or layer.stride!=(1,) or layer.padding!=(0,)
                    or layer.dilation!=(1,) or layer.groups!=1 or layer.padding_mode!='zeros'):
                raise ValueError('Only native pointwise pair convolutions may use direct input gradients')
            self.layers.append(name)
            layer.forward=self._forward(layer,layer.forward)
        if len(self.layers)!=4:raise ValueError('Expected four native pair pointwise convolutions')

    def _forward(self,layer,native):
        def forward(x):
            self.stats['calls']+=1
            if x.ndim!=3 or x.shape[0]!=1 or x.shape[1]!=layer.in_channels:
                raise ValueError('Expected native sparse pair layout (1,channels,pairs)')
            large=max(x.shape[1],layer.out_channels)*x.shape[-1]>INDEX_LIMIT
            if not (layer.training and torch.is_grad_enabled() and (large or self.force)):
                return native(x)
            if max(x.shape[-1],layer.in_channels,layer.out_channels)>INDEX_LIMIT:
                raise ValueError('Pointwise GEMM dimension exceeds int32')
            self.stats['fallback_calls']+=1
            self.stats['largest_pair_count']=max(self.stats['largest_pair_count'],x.shape[-1])
            weight,bias=layer.weight,layer.bias
            if torch.is_autocast_enabled(x.device.type):
                dtype=torch.get_autocast_dtype(x.device.type)
                x=x.to(dtype);weight=weight.to(dtype)
                if bias is not None:bias=bias.to(dtype)
            with torch.autocast(x.device.type,enabled=False):
                return _Pointwise.apply(x,weight,bias,self.stats)
        return forward


def validate_stats(stats,*,require_fallback=False):
    if (not isinstance(stats,dict) or set(stats)!={'calls','fallback_calls','input_gradient_calls','largest_pair_count'}
            or any(type(v) is not int or v<0 for v in stats.values()) or stats['calls']<=0
            or stats['fallback_calls']>stats['calls'] or stats['input_gradient_calls']>stats['fallback_calls']
            or (require_fallback and (stats['input_gradient_calls']==0 or stats['largest_pair_count']==0))):
        raise ValueError('Missing/invalid pointwise input-gradient evidence')


def validate_parities(proofs):
    from .pair_storage import POLICY as STORAGE_POLICY,PARITY_TOLERANCES,validate_storage_stats
    if not isinstance(proofs,list) or len(proofs)!=4:raise ValueError('Missing native and fallback parity')
    for proof,(precision,fallback) in zip(proofs,[('fp32',False),('bf16',False),('fp32',True),('bf16',True)]):
        if (proof.get('passed') is not True or proof.get('precision')!=precision or proof.get('steps')!=3
                or proof.get('policy')!=STORAGE_POLICY or proof.get('tolerance')!=PARITY_TOLERANCES[precision]
                or proof.get('forced_fallback_reference',False)!=fallback):
            raise ValueError('Native/fallback parity contract differs')
        validate_storage_stats(proof['statistics'],calls=3)
        if fallback:
            if proof.get('pointwise_policy')!=POLICY:raise ValueError('Parity gradient policy differs')
            validate_stats(proof['pointwise_statistics'],require_fallback=True)
