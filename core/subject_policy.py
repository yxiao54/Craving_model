OUD_SUBJECT_PREFIXES = ("88", "99")

TRAIN_ONLY = {
    "9915", "9915v2",
    "9933", "9933v2",
    "9945", "9945v2",
    "9973", "9973v2",
}

EXCLUDED_TASKS = {
    "good",
    "bad",
    "stress",
}


def summarize_subject_label_types(df, label_name):
    mixed_subjects = []
    single_class_subjects = []
    details = {}
    grouped = df.groupby(df["participant_id"].astype(str))
    for subject_id, group in grouped:
        labels = sorted(set(group[label_name].astype(int).tolist()))
        details[str(subject_id)] = labels
        if labels == [0, 1]:
            mixed_subjects.append(str(subject_id))
        elif labels in ([0], [1]):
            single_class_subjects.append(str(subject_id))
    return sorted(mixed_subjects), sorted(single_class_subjects), details


def is_oud_craving_subject(subject_id):
    subject_id = str(subject_id)
    return subject_id.startswith(OUD_SUBJECT_PREFIXES)


def filter_oudlab_subjects_for_label(df, label_name):
    df = df.copy()
    df["participant_id"] = df["participant_id"].astype(str)
    if label_name == "craving":
        df = df[df["participant_id"].map(is_oud_craving_subject)].copy()
    return df
