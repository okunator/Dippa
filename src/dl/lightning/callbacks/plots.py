# #  This is not working yet
 
 
#  def plot_metrics(self, scale: str = "log", metric: str = "loss", save:bool = False) -> None:
#         """
#         Plot the training and validation loss to same graph

#         Args:
#             conf (DictConfig): the config 
#             scale (str): y-axis scale. One of ("log", "normal").
#             metrics (str): One of the averaged metrics ("loss", "accuracy", "TNR", "TPR").
#             save (bool): Save result image
#         """
#         # cludge
#         assert scale in ("log", "linear"), "y-scale not in ('log', 'linear')"
#         assert metric in ("loss", "accuracy", "mean_iou"), "metric not in ('loss', 'accuracy', 'mean_iou')"
#         folder = f"{self.fm.experiment_dir}"
#         logdir = Path(f"{folder}/tf")
#         train_losses_all = {}
#         avg_train_losses_all = {}
#         avg_valid_losses_all = {}
#         avg_valid_accuracies_all = {}
#         avg_train_accuracies_all = {}
#         avg_valid_iou_all = {}
#         avg_train_iou_all = {}
#         epochs_all = {}

#         try:
#             train_losses_all = []
#             avg_train_losses_all = []
#             avg_valid_losses_all = []
#             avg_valid_accuracies_all = []
#             avg_train_accuracies_all = []
#             avg_valid_iou_all = []
#             avg_train_iou_all = []
#             epochs_all = []
#             summary_reader = SummaryReader(logdir, types=["scalar"])
#             for item in summary_reader:
#                 # print(item.tag)
#                 if item.tag == "train_loss":
#                     train_losses_all.append(item.value)
#                 elif item.tag == "epoch":
#                     epochs_all.append(item.value)
#                 elif item.tag == "avg_val_accuracy":
#                     avg_valid_accuracies_all.append(item.value)
#                 elif item.tag == "avg_train_accuracy":
#                     avg_train_accuracies_all.append(item.value)
#                 elif item.tag == "avg_val_loss":
#                     avg_valid_losses_all.append(item.value)
#                 elif item.tag == "avg_train_loss":
#                     avg_train_losses_all.append(item.value)
#                 elif item.tag == "avg_val_iou":
#                     avg_valid_iou_all.append(item.value)
#                 elif item.tag == "avg_train_iou":
#                     avg_train_iou_all.append(item.value)
#         except:
#             pass

#         np_train_losses = np.array(avg_train_losses_all)
#         np_valid_losses = np.array(avg_valid_losses_all)
#         np_valid_accuracy = np.array(avg_valid_accuracies_all)
#         np_train_accuracy = np.array(avg_train_accuracies_all)
#         np_valid_iou = np.array(avg_valid_iou_all)
#         np_train_iou = np.array(avg_train_iou_all)
#         np_epochs = np.unique(np.array(epochs_all))
        
#         df = pd.DataFrame({
#             "training loss": np_train_losses, 
#             "validation loss": np_valid_losses,
#             "training acc":np_train_accuracy,
#             "validation acc":np_valid_accuracy,
#             "training iou":np_train_iou,
#             "validation iou":np_valid_iou,
#             "epoch": np_epochs[0:len(np_train_losses)]
#         })

#         if metric == "accuracy":
#             y1 = "training acc"
#             y2 = "validation acc"
#         elif metric == "loss":
#             y1 = "training loss"
#             y2 = "validation loss"
#         elif metric == "mean_iou":
#             y1 = "training iou"
#             y2 = "validation iou"

#         plt.rcParams["figure.figsize"] = (20,10)
#         ax = plt.gca()
#         df.plot(kind="line",x="epoch", y=y1, ax=ax)
#         df.plot(kind="line",x="epoch", y=y2, color="red", ax=ax)
#         plt.yscale(scale)
        
#         if save:
#             plot_dir = logdir.parents[0].joinpath("training_plots")
#             plot_dir.mkdir(parents=True, exist_ok=True)
#             plt.savefig(Path(plot_dir / f"{scale}_{metric}.png"))
        
#         plt.show()

