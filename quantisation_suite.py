import torch
import copy


def run_quantisation_suite(model_label, base_model, evaluator, quant_funcs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on: {device.type.upper()} ---")

    base_model = base_model.to(device)
    models_to_test = {f"{model_label}_Baseline_FP32": base_model}

    for suffix, quant_fn in quant_funcs.items():
        print(f"Preparing {model_label}_{suffix}...")
        test_model = copy.deepcopy(base_model)

        try:
            test_model = quant_fn(test_model)
            models_to_test[f"{model_label}_{suffix}"] = test_model

        except Exception as e:
            print(f"   [Error] {suffix} failed: {e}")

    print(f"Sending {len(models_to_test)} models to the evaluator...")
    evaluator.evaluate_many(models_to_test)

    return evaluator
