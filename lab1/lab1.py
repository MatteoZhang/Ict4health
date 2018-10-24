import pandas as pd
import numpy as np

if __name__ == "__main__":
    x = pd.read_csv("parkinsons_updrs.data")
    x.info()
    x.describe()
    x.plot()
    realdata = x.values  # if there is () = method while without it 's an attribute
    np.random.shuffle(realdata)
    print("Matrix inside the file:\n", realdata)
    print("shape: ", np.shape(realdata))

    data = realdata[:, 4:21]
    print("Useful data :\n", data)
    print("shape:", np.shape(data))

    





