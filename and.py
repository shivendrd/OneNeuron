from utils.model import Perceptron
from utils.all_utils import prepare_data,save_plot, save_model
import numpy as np
import pandas as pd 


def main(data, eta, epochs, filename, plotFilename):
    """it will give the data send the data to dataframe create a plot creating a model 

    Args:
        data ([pd.DataFrame]): its the pandas dataframe
        eta ([type]): step size
        epochs ([epochs]): total no of iteration does
        filename ([model]): filename of model
        plotFilename ([plot]): name of plot which are created
    """

    df = pd.DataFrame(data)
    print(df)

    X,y = prepare_data(df)


    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    _ = model.total_loss

    save_model(model, filename=filename)
    save_plot(df, plotFilename, model)

if __name__=='__main__': #'<<<entry point'
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y":  [0,0,0,1],
    }

    ETA = 0.3
    EPOCHS = 10

    main(data=AND, eta=ETA, epochs=EPOCHS, filename="and.model", plotFilename="and.png")