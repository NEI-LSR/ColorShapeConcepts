# %%
import matplotlib.pyplot as plt
import numpy as np
from bin.dataloader import TrialDataLoader
from matplotlib import pyplot as plt
from bin import plotter

# %%
SUBJECT = "wooster"
CONTENT_ROOT = "/home/bizon/shared/isilon/PROJECTS/ColorShapeContingency1/MTurk1"
DATA_KEY_PATH = CONTENT_ROOT + "/subjects/" + SUBJECT + "/analysis/shape_color_attention_decodemk2_nohighvar_stimulus_response_data_key.csv"
IN_SETS = ['shape', 'color']
behavior_mode = [None, None]
if SUBJECT == 'wooster':
    bad = ["scd1_20240308", "scd2_20240308", "scd_20230813", ]
    test_good = None
elif SUBJECT == 'jeeves':
    bad = []
    test_good = None
abv_map = {"color": "colored_blobs",
           "shape": "uncolored_shapes",}

COLORS = ("LightRed", "DarkRed", "LightYellow", "DarkYellow", "LightGreen", "DarkGreen", "LightTurquiose",
          "DarkTurquiose", "LightBlue", "DarkBlue", "LightPurple", "DarkPurple", "LightGray", "DarkGray")
SHAPES = ("Hourglass", "UpArrow", "Diamond", "Spike", "Lock", "Bar", "Spade", "Dodecagon", "Sawblade", "Nail", "Rabbit", "Puzzle", "Venn", "Hat",)

# %%
# Define Set A and Set B
SetA = [2, 3, 6, 7, 10, 11]
SetB = [1, 4, 5, 8, 9, 12]

data = TrialDataLoader(
    DATA_KEY_PATH,
    1,
    set_names=list(abv_map.values()),
    content_root=CONTENT_ROOT, ignore_class=[],
    cv_folds=1, ignore_sessions=bad,
    mask=None, seed=42, cube=False, start_tr=1, end_tr=3,
    behavior_mode=behavior_mode,
    override_use_sess=None)


# %%
def compute_class_accuracy(df, true_col, correct_col, include_col):
    """
    Computes class-wise accuracy for each unique label.
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    true_col (str): Column name for the true labels.
    correct_col (str): Column name indicating whether the prediction was correct (1) or incorrect (0).
    include_col (str): Column name indicating whether the example should be included (boolean or binary).

    Returns:
    dict: A dictionary mapping each unique label to its class accuracy.
    """
    # Filter the dataframe based on the include_col
    filtered_df = df[df[include_col] == 1]

    # Group by true label and compute accuracy
    accuracy_dict = (
        filtered_df.groupby(true_col)[correct_col]
        .mean()
        .to_dict()
    )

    return accuracy_dict

cdata = data.folded_sets["colored_blobs"].get_all()
sdata = data.folded_sets["uncolored_shapes"].get_all()
data = [sdata, cdata]
fig, ax = plt.subplots(2)
types = ["shape", "color"]
type_labels = [SHAPES, COLORS]
for i in range(2):
    accs = compute_class_accuracy(data[i], "condition_name", "correct", "make_choice")
    labels = type_labels[i]
    accs = np.array([accs[l] for l in labels])[:, None, None]
    plotter.create_save_barplot(ax[i], fig, SUBJECT + "_" + types[i] + "_active_task_mri_classwise_accuracy", accs,
                                labels, out_dir="../figures/other", ymax=1.0)
fig.show()

print("done!")