import copy
import os

savePath = "../../dataset/test.txt"
new_test=[]
error_test = []
with open(savePath, "r") as f:
    for line in f:
        orgin_line = copy.copy(line)
        line = line.strip()
        line = line.replace("hy-tmp/", '')
        if os.path.exists(os.path.join(line, 'rgb')):
            print("1")
            new_test.append(orgin_line)
        else:
            print("2")
            error_test.append(orgin_line)

    f.close()

with open("../../dataset/new_test.txt", "w") as f:
    f.writelines(new_test)
    f.close()

with open("../../dataset/error_test.txt", "w") as f:
    f.writelines(error_test)
    f.close()


