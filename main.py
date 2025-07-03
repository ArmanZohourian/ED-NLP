import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

labels = ["anger", "fear", "joy", "sadness", "surprise"]

def predict_single_text(
        text: str,
        checkpoint_dir: str = "output_dir/checkpoint-695",
        threshold: float = 0.5
) -> list[str]:
    """

    The used default checkpoint directory is hardcoded from running the jupyter notebook used
    for fine-tuning the model. Specify a different checkpoint directory if wanted.

    :param text: An element of the 'text' column of the track-a.csv dataframe
    :param checkpoint_dir: the model checkpoint from training using the 'transformers' library
    :param threshold: needed probability for an emotion to apply

    :return: A list of strings containing the predicted emotions, as defined in track-a.csv
    """

    tok  = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).eval()

    # multi-label uses sigmoid, not softmax
    probs = model(**(tok(text, return_tensors="pt"))).logits.squeeze(0).sigmoid()

    return [emotion for prob, emotion in zip(probs, labels) if prob.item() >= threshold]

def predict(csv_file: str) -> list[list[str]]:
    """
    :param csv_file: Expect a path to a csv file in the same format as track-a.csv

    :return: the predicted emotions per row of the dataframe
    """

    df = pd.read_csv(csv_file)
    return [predict_single_text(x) for x in (df['text'])]

if __name__ == '__main__':
    result = predict("track-a-head.csv")
    print(result)
