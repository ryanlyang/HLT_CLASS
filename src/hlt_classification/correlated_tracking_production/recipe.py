"""Production authorization selects one unchanged historical pilot endpoint."""
from hlt_classification.correlated_tracking.kernel import recipe as pilot_recipe
from hlt_classification.literature_context.transform import build_inputs
from .contracts import artifact


def input_contract():
    return artifact('INPUTS', dimensions=17, tracking_columns=[11, 12, 13, 14],
        value_transform='asinh(value/[0.1,0.2])/4',
        error_transform='log1p(error/[0.02,0.05])/2', tracking_clipping=False,
        geometry='unchanged_own_view', required_for_all_comparison_arms=True,
        metadata_is_not_features=True)


def recipe():
    # The pilot recipe's no-training flag describes its original authorization.
    # Never rewrite it: this new wrapper authorizes the selected production arm.
    return artifact('RECIPE', parents={'pilot_recipe': pilot_recipe()['content_hash']},
        name='CORR_MID', kernel=pilot_recipe(), strength=1., mode='CORR',
        input_contract=input_contract(), replica=0, redraw_per_epoch=False,
        strength_selection='explicit_user_choice_before_classifier_results',
        test_materialization_authorized=True, test_inference_authorized=False,
        classification='synthetic_mechanism_not_CMS_HLT')
