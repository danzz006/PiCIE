import re
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np


def smooth(y):
    x = np.array([*range(len(y))])
    X_Y_Spline = make_interp_spline(x, y)
 
    # Returns evenly spaced numbers
    # over a specified interval.
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    
    return X_, Y_


log_file_path = r"C:\Users\PC\Desktop\train.log"
dest_dir_path = "segmentation_results"

with open(log_file_path, "r") as fs:
    logs = fs.read()
    
    
pattern_acc_final = re.compile(r"ACC: [0-9]+.[0-9]+")
pattern_acc_all = re.compile(r"ACC  - All:+.[0-9]+.[0-9]+")
pattern_acc_thing = re.compile(r"ACC  - Thing:+.[0-9]+.[0-9]+")
pattern_acc_stuff = re.compile(r"ACC  - Stuff:+.[0-9]+.[0-9]+")

pattern_miou_final = re.compile(r"mIoU: [0-9]+.[0-9]+")
pattern_miou_all = re.compile(r"mIOU - All:+.[0-9]+.[0-9]+")
pattern_miou_thing = re.compile(r"mIOU - Thing:+.[0-9]+.[0-9]+")
pattern_miou_stuff = re.compile(r"mIOU - Stuff:+.[0-9]+.[0-9]+")


acc_final = pattern_acc_final.findall(logs)
acc_all = pattern_acc_all.findall(logs)
acc_thing = pattern_acc_thing.findall(logs)
acc_stuff = pattern_acc_stuff.findall(logs)

acc_final = [float(i.split(":")[-1]) for i in acc_final]
acc_all = [float(i.split(":")[-1]) for i in acc_all]
acc_thing = [float(i.split(":")[-1]) for i in acc_thing]
acc_stuff = [float(i.split(":")[-1]) for i in acc_stuff]

# acc_supervised = [acc_final[i] for i in range(len(acc_final)) if i % 4 == 0]
# acc_unsupervised = [acc_final[i] for i in range(len(acc_final)) if i % 4 != 0]

acc_supervised = [acc_final[i] for i in range(len(acc_final))]
acc_unsupervised = [acc_final[i] for i in range(len(acc_final))]

acc_supervised_X, acc_supervised_Y = smooth(acc_supervised[:50])
acc_unsupervised_X, acc_unsupervised_Y = smooth(acc_unsupervised)


acc_final_X, acc_final_Y  = smooth(acc_final[:50])
# acc_all_X, acc_all_Y =  smooth(acc_all)
# acc_thing_X, acc_thing_Y  = smooth(acc_thing)
# acc_stuff_X, acc_stuff_Y  = smooth(acc_stuff)



miou_final = pattern_miou_final.findall(logs)
miou_all = pattern_miou_all.findall(logs)
miou_thing = pattern_miou_thing.findall(logs)
miou_stuff = pattern_miou_stuff.findall(logs)

miou_final = [float(i.split(":")[-1]) for i in miou_final]
miou_all = [float(i.split(":")[-1]) for i in miou_all]
miou_thing = [float(i.split(":")[-1]) for i in miou_thing]
miou_stuff = [float(i.split(":")[-1]) for i in miou_stuff]


# miou_supervised = [acc_final[i] for i in range(len(miou_final)) if i % 4 == 0]
# miou_unsupervised = [acc_final[i] for i in range(len(miou_final)) if i % 4 != 0]

miou_supervised = [miou_final[i] for i in range(len(miou_final))]
miou_unsupervised = [miou_final[i] for i in range(len(miou_final))]

miou_supervised_X, miou_supervised_Y = smooth(miou_supervised[:50])
miou_unsupervised_X, miou_unsupervised_Y = smooth(miou_unsupervised[:50])


miou_final_X, miou_final_Y  = smooth(miou_final)
# miou_all_X, miou_all_Y  =  smooth(miou_all)
# miou_thing_X,miou_thing_Y   = smooth(miou_thing)
# miou_stuff_X, miou_stuff_Y  = smooth(miou_stuff)



# plt.plot(acc_unsupervised_X, acc_unsupervised_Y, label="ACC")
# plt.plot(acc_all, label="ACC - ALL")
# plt.plot(acc_thing, label="ACC - THING")
# plt.plot(acc_stuff, label="ACC - STUFF")
# plt.title("ACCURACIES")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig(dest_dir_path+"/acc_unsupervised.png")

# plt.clf()

# plt.plot(miou_final, label="mIOU")
# plt.plot(miou_unsupervised_X,miou_unsupervised_Y, label="mIOU")
# plt.plot(miou_thing_X, miou_thing_Y, label="mIOU - THING")
# plt.plot(miou_stuff_X, miou_stuff_Y, label="mIOU - STUFF")
# plt.title("mIOU Unupervised")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig(dest_dir_path+"/miou_unsupervised.png")
# print("figures saved..")


def save_miou_supervised():
    plt.clf()
    plt.plot(miou_supervised_X,miou_supervised_Y, label="mIOU")
    plt.title("mIOU")
    plt.legend()
    plt.savefig(dest_dir_path+"/miou_supervised.png")
    print("figure saved..")
    
def save_miou_unsupervised():
    plt.clf()
    plt.plot(miou_unsupervised_X,miou_unsupervised_Y, label="mIOU")
    plt.title("mIOU Unsupervised")
    plt.legend()
    plt.savefig(dest_dir_path+"/miou_unsupervised.png")
    print("figure saved..")
    
    
def save_acc_supervised():
    plt.clf()
    plt.plot(acc_supervised_X, acc_supervised_Y, label="ACC")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(dest_dir_path+"/acc_supervised.png")
    print("figure saved..")

def save_acc_unsupervised():
    plt.clf()
    plt.plot(acc_unsupervised_X, acc_unsupervised_Y, label="ACC")
    plt.title("Accuracy Unsupervised")
    plt.legend()
    plt.savefig(dest_dir_path+"/acc_unsupervised.png")
    print("figure saved..")




if __name__ == '__main__':
    save_miou_supervised()
    save_miou_unsupervised()
    save_acc_supervised()
    save_acc_unsupervised()
        