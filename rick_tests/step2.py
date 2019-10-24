from deepposekit import Annotator

app = Annotator(datapath=r'C:\Users\rick\Desktop\DeepPoseKit_test\annotation_set.h5',
                dataset='images',
                skeleton=r'C:\Users\rick\Desktop\DeepPoseKit_test\skeleton.csv',
                shuffle_colors=False,
                text_scale=0.4,
                scale=2)
app.run()