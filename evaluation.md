# evaluation

> Below is the raw text to added and formatted into the milestone 4 report.

---

### test dataset

The test dataset contains 116 rows, which is 4 percent of the 2768 rows provided in the training dataset.
It has the exactly same prediction categories as defined in the training dataset, which are: anger, fear, joy, sadness, surprise.

As shown in the introduction, most of the emotion labels of the training data are imbalanced, the strongest imbalance being *anger* at only about 12 percent.
After that, *joy*, *sadness* and *surprise* are all in the range of 25 to 30 percent, and *fear* is closest to being balanced
with 58 percent.

Assuming the test dataset is distributed similarly, one can expect only 0.12 * 116 = 14 positive instances of anger in the test set, which are concerningly few.
As the test set it is fixed, we proceed and keep that limitation in mind.

Checking the value distribution in the test set, we see that the actual values are close to these estimations.
The labels with the biggest difference are *fear* and *surprise*, which are overrepresented in the test data by 3.9 and 3.6 percentage points respectively.

### model evaluation

We inspect F1-Score per label, F1-Score aggregated and accuracy.

The accuracies per label for the test set are similar to the training. The biggest difference is for *sadness* with the
accuracy of 76 percent during training, but 86 for the test set. We can see this by checking the standard deviation of the other 4 differences, which
is only 2.4 percentage points.

In training, we archived 79% accuracy average across all predicted fields, and 34% of rows with all 5 labels predicted correctly.
For the test set, we beat these numbers with total field accuracy of 81% and perfect row accuracy of 36 percent.
Usually, one can expect a worse performance on the test set, but we attribute these findings to the limited size of the test data,
as mentioned in the introduction.

The micro-average F1-Score on the test set is 0.66, compared to 0.63 in training.
TODO check per label F1 score.