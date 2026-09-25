"""Valid-coordinate accounting, separate from inactive placeholder clipping."""
import numpy as np

from .b_tracking_generation import MASKS, COORDINATES
from .c_diagnostic_metrics import Counters, accumulate, moments


class TrackingCounters(Counters):
    def add(self, info, variant):
        super().add(info)
        coordinates, cells = {}, {}
        for event in info["events"]:
            scope = event["module"]+"/"+event["mechanism"]
            state = event["state"]
            key = scope+"/state"+",".join(map(str, state))+"/"+event.get(
                "selected_residual_cell", "identity_fallback")
            cells[key] = cells.get(key, 0)+1
            if "response" not in event:
                continue  # Fallbacks stay in original counters and physical histograms.
            before, after = event["response"]
            low, high = np.broadcast_arrays(*np.asarray(event["response_limits"]))
            low, high = np.broadcast_to(low, before.shape), np.broadcast_to(high, before.shape)
            for j in range(len(before)):
                axis, particle = j % 8, j // 8
                if axis < 4:
                    continue
                pid, mask = int(state[4*particle]), int(state[4*particle+2])
                valid = bool(mask & (1 << (axis-4)))
                key = f"{scope}/pid{pid}/mask{mask}/{COORDINATES[axis]}"
                item = dict(applicable=int(valid), inapplicable=int(not valid))
                if valid:
                    a, b, central = float(before[j]), float(after[j]), float(event["central"][j])
                    item.update(central=moments([central]), raw_increment=moments([event["raw_increment"][j]]),
                        applied_increment=moments([event["applied_increment"][j]]), before=moments([a]), after=moments([b]),
                        absolute_correction=moments([abs(a-b)]), low_clipped=int(a < low[j]),
                        high_clipped=int(a > high[j]), central_outside=int(central < low[j] or central > high[j]),
                        residual_available=int(event["residual_available"]),
                        residual_disabled=int(event["residual_available"] and axis in MASKS[variant]),
                        scale_evaluated=int("scale" in event),
                        scale_clipped=int("scale" in event and event["scale"][0][j] != event["scale"][1][j]))
                accumulate(coordinates.setdefault(key, {}), item)
        accumulate(self.value, dict(tracking_coordinates=coordinates, selected_residual_cells=cells))


def model_envelopes(response):
    """Model inventory, not evaluation occupancy; error coordinates are logarithmic."""
    return {name: dict(response_limits=model["response_limits"], scale_limits=model["scale_limits"],
                      coordinate_cycle=list(COORDINATES),
                      backend_states=[row["state"] for row in model["backends"]])
            for name, model in response["modules"].items() if name.endswith("_value")}
