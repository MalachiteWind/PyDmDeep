import subprocess

def execute_command(command):
    """
    Executes a command in the shell and returns its status and output.
    
    Args:
        command (str): The command to execute.
    
    Returns:
        tuple: (status, output), where status is True if successful, and output is the stdout or error.
    """
    try:
        result = subprocess.run(command, shell=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)

def main():
    base_command = (
        "nohup mitosis data lstm dmd -F trials/dmd_results --debug "
        "-e data.seed=1234 "
        "-e data.lags=50 "
        "-e data.train_len={train_len}"
        "-e data.rand=False "
        "-e data.scaler=\"std\" "
        "-e data.target_is_statespace=False "
        "-e data.k_modes=2 "
        "-p lstm.opt=ADAM "
        "-p lstm.loss=MSE "
        "-p lstm.dataloader_kws=small_batch "
        "-e lstm.hidden_size=64 "
        "-e lstm.num_layers=2 "
        "-p lstm.num_epochs=1k "
        "-e lstm.seed=1234 &> out_docs/dmd_results{train_len_file}.out &"
    )

    # Range of train_len values to iterate over
    train_len_values = [
        0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,
        0.95
    ]

    for train_len in train_len_values:
        # Prepare the command with updated train_len
        train_len_formatted = f"{train_len:.2f}"  # Format for display (e.g., 0.75)
        train_len_file = int(train_len * 100)    # File-friendly integer (e.g., 75)
        command = base_command.format(train_len=train_len_formatted, train_len_file=train_len_file)

        print(f"Executing command with train_len={train_len_formatted}...")
        status, output = execute_command(command)
        if status:
            print(f"Successfully executed for train_len={train_len_formatted}.\n")
        else:
            print(f"Failed to execute for train_len={train_len_formatted}. Error:\n{output}\n")

if __name__ == "__main__":
    main()
