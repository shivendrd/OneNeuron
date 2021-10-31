from utils.model import Perceptron
from utils.all_utils import prepare_data,save_plot, save_model
import numpy as np
import pandas as pd 

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y":  [0,0,0,1],
}

# X,y = prepare_data(AND)

df = pd.DataFrame(AND)
print(df)

X,y = prepare_data(df)
ETA = 0.3

EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X,y)

_ = model.total_loss

save_model(model, filename="and.model")
save_plot(df, "and.png", model)