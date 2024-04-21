from vis_get_QE import main as main_vis

model_path = "result/stage2/model_best.pth.tar"  # path to the folder containing the model_best.pth.tar file
data_path = "data/CIFAR10/"  # path to the folder containing the CIFAR10 dataset

# extract freq from result/wandb/7mniaeph/config.txt
with open(model_path.replace("model_best.pth.tar", "config.txt"), "r") as file:
    lines = file.readlines()

    for line in lines:
        # Split the line into key-value pairs
        parts = line.strip().split(':')
        if len(parts) == 2:
            key, value = parts[0].strip(), parts[1].strip()

            # Check if the key is exactly 'freq'
            if key == 'model':
                model_name = value
                break  # Stop reading further as we found our value

    for line in lines:
        # Split the line into key-value pairs
        parts = line.strip().split(':')
        if len(parts) == 2:
            key, value = parts[0].strip(), parts[1].strip()

            # Check if the key is exactly 'freq'
            if key == 'freq':
                freq_value = float(value)
                break  # Stop reading further as we found our value


print(f"Model path: {model_path}")
print(f"Data path: {data_path}")
print("Model name: ", model_name)
print("Frequency value: ", freq_value)
QE, b = main_vis(model_path, ckpt_freq=freq_value, model=model_name, data_path=data_path)
