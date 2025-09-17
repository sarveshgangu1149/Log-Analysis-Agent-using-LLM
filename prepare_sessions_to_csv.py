import pandas as pd

# Path to your sessions.txt file
sessions_txt = "dataset/sessions.txt"
output_csv = "dataset/messages_sessions.csv"

labels = [1, 1]

def parse_sessions(file_path):
    sessions = []
    current_session = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("session"):
                if current_session:
                    sessions.append(current_session)
                    current_session = []
            else:
                current_session.append(line)
        if current_session:
            sessions.append(current_session)
    return sessions

sessions = parse_sessions(sessions_txt)

if len(labels) != len(sessions):
    raise ValueError("Number of labels does not match number of sessions.")

df = pd.DataFrame({
    "Content": [" ;-; ".join(session) for session in sessions],
    "Label": labels
})

df.to_csv(output_csv, index=False)
print(f"CSV saved to {output_csv}")
