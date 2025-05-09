import pandas as pd

def apply_demographic_mapping(user_df: pd.DataFrame, mapping_path: str) -> pd.DataFrame:
    """mapping_categories.csv 파일을 바탕으로 user_df에 *_idx 컬럼을 추가한다."""
    mapping_df = pd.read_csv(mapping_path)
    mapping_dict = {row["category"]: row["index"] for _, row in mapping_df.iterrows()}

    user_df = user_df.copy()

    user_df["gender_idx"] = user_df["gender"].map(lambda g: mapping_dict[f"gender_{g}"])
    user_df["age_idx"] = user_df["age"].map(lambda a: mapping_dict["age_25_and_under"] if a <= 25 else mapping_dict["age_26_and_over"])
    user_df["major_idx"] = user_df["major"].map(lambda m: mapping_dict.get(f"major_{m}", mapping_dict["major_other"]))
    user_df["grade_idx"] = user_df["grade"].map(lambda g: mapping_dict.get(f"grade_{g}", mapping_dict["grade_eternal"]))

    return user_df
