import subprocess

for epoch in range(50):
    print("epoch", epoch)
    subprocess.run(["python", "train_and_eval.py"])