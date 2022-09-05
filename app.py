from transformers import pipeline, set_seed
import gradio as gr

classifier = pipeline('text-generation', model='gpt2')
set_seed(42)

def generate_text(text, gen_length):
    gen_text = classifier(text, max_length=gen_length)[0]['generated_text']
    return gen_text

Instructuction = "Browse the internet to download any unique image"
title="Text generation playground"
description = "Start writting a peice of text in the input box\
               and see how well the text generation language model\
               is able to generate new text that uniquely completes your sentences."
article = """
            - Write a text in the input box and specify the length of text.
            - Also you can select a quick example to continue.
            - Click submit button to generate new text.
            - Click clear button to try new text generation.
          """

# Gradio app design
interface = gr.Interface(
            generate_text,
            inputs = ['text', gr.Slider(20, 120, value=80, step=1)],
            outputs='text',
            title = title,
            description = description,
            article = article,
            allow_flagging = "never",
            #theme = "peach",
            #live = False,
            examples=[["Agriculture is very fundamental to",
                      50], ["I will tell a story about",
                      100]]
            )
interface.launch()
