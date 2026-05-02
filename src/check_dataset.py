import os

base_path = "Multi class/train/drowsy/yawning"

subjects = set()
conditions = set()
statuses = set()
sequences = set()
ranges = {}

for file in os.listdir(base_path):
    if file.endswith(".jpg"):
        parts = file.split("_")

        if len(parts) >= 5:
            subject = parts[0]
            condition = parts[1]
            sequence = parts[2]   #  harusnya tetap
            timeframe = parts[3]
            status = parts[4].replace(".jpg", "")

            subjects.add(subject)
            conditions.add(condition)
            sequences.add(sequence)
            statuses.add(status)

            # key gabungan
            key = (subject, condition, sequence)

            try:
                tf = int(timeframe)
            except:
                continue

            # update min/max
            if key not in ranges:
                ranges[key] = {"min": tf, "max": tf}
            else:
                ranges[key]["min"] = min(ranges[key]["min"], tf)
                ranges[key]["max"] = max(ranges[key]["max"], tf)


# 🔹 OUTPUT
print("Subjects:", subjects)
print("Conditions:", conditions)
print("Statuses:", statuses)

print("\nRange per subject-condition-sequence:")
for (sub, cond, seq), val in sorted(ranges.items()):
    print(f"{sub}_{cond}_{seq} -> start={val['min']}, end={val['max']}")