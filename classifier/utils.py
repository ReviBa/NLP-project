def freeze_bert_top_layers(model, from_layer=11, debug=False):
    top_layer_params = []
    if debug:
        print("Frozen layers:")
    for name, para in model.named_parameters():
        if name.startswith("bert") and not name.startswith(f"bert.encoder.layer.{str(from_layer)}"):
            para.requires_grad = False
        else:
            if debug:
                print(name)
            top_layer_params.append(para)
    return top_layer_params
