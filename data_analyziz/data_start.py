import matplotlib.pyplot as plt
import xml.etree.cElementTree as etree
import os

data_type = {"PI": 0, "PN": 0, "XO": 0, "NP": 0, "HD": 0, "FB": 0, "FO": 0, "FP": 0}
annotation_path="/Users/crispus/Desktop/capstone/VOC2007/Annotations/"

def readfile():
    train_data = open("/Users/crispus/Desktop/capstone/VOC2007/ImageSets/Main/test.txt")
    lines = train_data.readlines()
    for line in lines:
        data_type[line[0:2]]+=+1  # Python [0:2] will not include [2]
    print(data_type)

def check_num_flaw():
    """""
       Check the number of flaw in each photos
    """
    anotations=os.listdir(annotation_path)
    record=[]
    for annotation in anotations:
        with open(annotation_path+annotation) as at:
            elem = etree.parse(at)
            flaws=elem.findall("object")
            record.append(len(flaws))
    print(record)

def draw_picture():
    lable=data_type.keys()
    val=list(data_type.values())
    bars=plt.bar(range(len(data_type)),height=val)
    plt.xticks(range(8),lable)
    for bar in bars :
        height=bar.get_height()
        plt.text(bar.get_x()+0.4,height+1, str(height), ha="center", va="bottom")
    plt.xlabel("data type")
    plt.title("testing data analysis")
    # using plt.savefig before plt.show,because plt.show will return blank.
    plt.savefig("/Users/crispus/Desktop/capstone/testdata.png")
    plt.show()

if __name__ == '__main__':
    check_num_flaw()