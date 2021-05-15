from keras.models import load_model

# Load model
model = load_model('Reddit_comments_model_2.h5')

def predict_class(Comment):
    
    
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len=50
    text = []
    text.append(Comment)
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    return sentiment_classes[yt[0]]
    
    
import gradio as gr


iface = gr.Interface(fn=predict_class, inputs="text", outputs="text")
iface.launch()