import gradio as gr
import pandas as pd
import numpy as np
import os
import re
import html
from collections import OrderedDict

### define HTML components ###
##############################
HTML_P = """<p style="text-align:left;font-family:Tahoma;font-size:16px">[TEXT_FILL]</p>"""
HTML_SPAN = """<span style="color:[COLOR_FILL];"><b>[TEXT_FILL]</b></span>"""
C1, C2, EX = 'CornflowerBlue', 'BlueViolet', 'ForestGreen'
part = "part3B"


# Functions for wrapping text with formatting
def wrap_span(color: str, text: str) -> str:
    return HTML_SPAN.replace('[COLOR_FILL]', color).replace('[TEXT_FILL]', text)


def wrap_observations(obs1: str) -> str:
    obs1 = obs1.replace("�", r"'").replace("\n", "<br>")
    # obs1_html = wrap_span(C1, 'Post: ') + obs1
    return HTML_P.replace('[TEXT_FILL]', obs1 + '<br><br><hr>')


HTML_H2 = """<h3 style="text-align:center;font-family:Tahoma;font-size:24px;text-decoration:underline">[TEXT_FILL]</h3>"""


def wrap_title(title: str) -> str:
    title = title.replace("�", r"'").replace("\n", "<br>")
    return HTML_H2.replace('[TEXT_FILL]', title)


# HTML strings for our instructions
start_html = f"""<p style="text-align:left;font-family:Tahoma;font-size:16px">
                <span style="color:IndianRed;"><b>Please enter your name and click next.</b></span><br>
                The name should contain only English letters, numbers or spaces. During annotation, do not change it.

                <p style="text-align:left;font-family:Tahoma;font-size:16px">
                       <span style="color:IndianRed;"><b>Instructions {part}.</b></span><br>
                       During annotation, do not change your name.<br>
                       In each round, You will be presented one Post.<br><br>
                       <b>Task Description:</b><br>
                        The annotation task aims to assess the risk level of the relationship described in the Reddit post. As an annotator, your task is to read each post carefully and answer the provided questions based on the content of the post.<br><br>
                        <b>Guidelines:</b><br>
                        <ul>
                          <b> 1. Carefully read the entire Reddit post before answering the questions. </b><br>
                          <b> 2. Keep in mind the task's objective to assess the risk level of the relationship.</b><br>
                          <b> 3. Consider the post's language and tone when answering the questions.</b><br>
                          <b> 4. Try to understand the context of the post and the relationship dynamics between the individuals involved.</b><br>
                          <b> 5. Answer the questions based solely on the information presented in the post.</b><br>
                          <b> 6. Try not to let personal biases or assumptions affect your annotations.</b><br>
                          <b> 7. Each post will have a separate set of questions that you need to answer.</b><br>
                        </ul>

"""

end_html = """<p style="text-align:left;font-family:Tahoma;font-size:16px">
              <span style="color:IndianRed;"><b>Thank you for participating!</b></span><br>
              You may close the page now :)</p>"""
instruction_html = f"""<p style="text-align:left;font-family:Tahoma;font-size:16px">
                       <span style="color:IndianRed;"><b>Instructions {part}:</b></span><br>
                       During annotation, do not change your name.<br>
                       In each round, You will be presented with one {wrap_span(C1, 'Post')}.<br>
                       Your task is to read the post very carefully, and to answer the ~20 questions afterwards. <br>
                       Answer <b> yes </b> if you think the described situation matches the question, <b> no </b> if it doesn't. <br><br>
                       
                       <span style="color:IndianRed;"><b>Special cases:</b></span><br>
                       If you need to <b> skip </b> the post ----> enter <b> 1 </b> in both ages boxes. <br>
                       If there is  <b> no age </b> mention for one of the persons in the posts ----> enter <b> 100 </b> in the relevant age box.
                       </p><br>"""

############# user and data generator ##############
####################################################

def get_example_id(num_examples: int, all_annotate_rounds: int, total_rounds: int, user_id: int, round_id: int) -> int:
    indices = list(range(num_examples))

    if round_id >= total_rounds:
        return indices[-1]

    if round_id < all_annotate_rounds:
        return indices[round_id]
    else:
        if user_id == 0:
            indices_new = list(range(90, 110))
        elif user_id == 1:
            indices_new = list(range(100, 120))
        elif user_id == 2:
            indices_new = list(range(90, 100)) + list(range(110, 120))
        else:
            indices_new = list(range(120, num_examples))

        updated_round_id = round_id % all_annotate_rounds

        if updated_round_id > len(indices_new):
            indices_new = list(range(num_examples))

        return indices_new[updated_round_id]


# This function returns the observations and explanation HTML strings.
def prepare_round(data: pd.DataFrame,  all_annotate_rounds: int, total_rounds: int, user_id: int, round_id: int, raw=False) -> tuple:
    # This function returns the index of the example the user should annotate (`example_id`).
    # Notice that we want half of the user's examples to overlap with half of previous user's examples.
    # Example 1: if we have 300 examples and each user annotates 100, then user1=[0-99], user2=[50-149], user3=[100-199],...
    # Example 2: if we have 300 examples and each user annotates 200, then user1=[0-199], user2=[100-299], user3=[200-299]+[0-99],...
    # The `example_id` is the index at `round_id`.
    example_id = get_example_id(data.shape[0], all_annotate_rounds, total_rounds, user_id, round_id)
    row = data.iloc[example_id]
    observations = wrap_observations(row['text'])
    title = wrap_title(row['title'])
    if raw:
        observations = row['text']
        title = row['title']
    return example_id, observations, title

# This function checks if the name is valid and if the user is new, if the user is not new, it returns his current round_id.
def get_user_info(annotations: pd.DataFrame, name: str) -> tuple:
    names = set(annotations['name'])
    # name is not a valid name:
    if name == '' or not bool(re.match("^[a-zA-Z0-9 ]*$", name)):
        return None, None, None
    # first user
    elif len(names) == 0:
        return name, 0, -1
    # new user but not first user
    elif name not in names:
        user_id = annotations['user_id'].max() + 1
        return name, user_id, -1
    # old user
    user_annotations = annotations[annotations['name'] == name]
    assert len(set(user_annotations['user_id'])) == 1
    user_id = user_annotations['user_id'].max()
    prev_round_id = user_annotations['round_id'].max()
    return name, user_id, prev_round_id + 1

############## prepare questions #########
##########################################
questions_df = pd.read_csv("data/raw/questions_v3.csv")
questions_ratings_filtered = OrderedDict({})
ages_questions = []
for i in range(len(questions_df)):
    question = questions_df.iloc[i, 0]
    answers =  questions_df.iloc[i, 1:]
    filtered_answers = [a for a in answers if type(a) == str]
    if len(filtered_answers) > 0:
        questions_ratings_filtered[question] = filtered_answers
    else:
        ages_questions.append(question)


annotations_path = os.path.join("data/annotated_data", f"annotations_{part}.csv")
# read data file
data_path = os.path.join("data", f"posts_abuse_shuffled.csv")
df = pd.read_csv(data_path)
df = df[df.part == part]
num_examples = df.shape[0]
all_annotate_rounds = 90
total_rounds = 110

print(num_examples)
n_questions_ratings = list(questions_ratings_filtered.keys())
cols = ['name', 'user_id', 'round_id', 'example_id', "curr_title", "curr_example"]
cols += [f'question_{i}_{q}' for i, q in enumerate(n_questions_ratings)]
cols += [f'question_{i+len(n_questions_ratings)}_{q}' for i, q in enumerate(ages_questions)]

# read/prepare annotations file
if os.path.exists(annotations_path):
    annotations = pd.read_csv(annotations_path)
else:
    annotations = pd.DataFrame(columns=cols)
    annotations.to_csv(annotations_path, index=False)

######### main function ##############
######################################
with gr.Blocks() as block:
    # define the components of the block

    instruction = gr.HTML(start_html, visible=True)
    name_box = gr.Textbox(lines=1, label="Please enter your name:", visible=True)
    title = gr.HTML('', visible=True)
    observations = gr.HTML('', visible=True)
    next_button = gr.Button(f"Next [0/{total_rounds}]")
    numbers = [gr.Number(label=question) for question in ages_questions]
    radios = [gr.Radio(rating_scale, label=question, value=rating_scale[0]) for question, rating_scale in
              questions_ratings_filtered.items()]

    # define the processing function
    def annotation_round(name, *args):
        global annotations
        args = list(args)
        ages = args[:len(numbers)]
        rates = args[len(numbers):]
        # get the user info - the user_id and his current round
        name, user_id, current_round_id = get_user_info(annotations, name)
        next_round_id = current_round_id + 1
        print(f"Name: {name} | user_id: {user_id} | Round: {current_round_id}")
        # prepare the new values of the components
        new_instruction = instruction_html
        new_observations = ''
        new_title = ''
        new_button = f"Next [{next_round_id}/{total_rounds}]"

        # name is invalid
        if name is None:
            new_instruction = start_html

        # finish
        elif current_round_id == total_rounds:
            new_instruction = end_html
            new_button = f"Done [{current_round_id}/{total_rounds}] :)"

        # valid name, user did not finish to annotate
        else:
            example_id = None
            save_annotations = False
            # new user - define rate to be None and save it. In the next round we will know the user is not new.
            if current_round_id == -1:
                example_id, save_annotations = None, True
                rates = [None for _ in range(len(radios))]
                ages = [None for _ in range(len(numbers))]
                new_button = f"Next [0/{total_rounds}]"

            # empty rate - do not save annotation - keep the current round.
            elif any(ele in ['', None, 0, 0.0] for ele in ages) or any(ele < 0 for ele in ages):
                example_id, save_annotations = None, False
                next_round_id = current_round_id
                new_button = f"Next [{current_round_id}/{total_rounds}] *Missing answers for ages (should be > 0)*"

            # valid annotation round.
            elif all(rates):
                example_id = get_example_id(num_examples, all_annotate_rounds, total_rounds, user_id, current_round_id)
                save_annotations = True

            # save annotations
            if save_annotations:
                example_id, curr_observations, curr_title = 0, "", ""
                if current_round_id >= 0:
                    example_id, curr_observations, curr_title = prepare_round(df, all_annotate_rounds, total_rounds, user_id, current_round_id, raw=True)

                row = OrderedDict({'name': name, 'user_id': user_id,
                                   'round_id': current_round_id, 'example_id': example_id, "curr_title": curr_title,
                                   "curr_example": curr_observations})
                i = 0
                for question, rad in zip(questions_ratings_filtered.keys(), rates):
                    row[f"question_{i}_{question}"] = rad
                    i += 1
                for question, num in zip(ages_questions, ages):
                    row[f"question_{i}_{question}"] = num
                    i += 1

                annotations = pd.concat([annotations, pd.DataFrame(row, index=[0])], ignore_index=True)
                annotations.tail(1).to_csv(annotations_path, header=None, mode='a', index=False)

            # prepare next round
            if next_round_id == total_rounds:
                new_instruction = end_html
                new_button = f"Done [{next_round_id}/{total_rounds}] :)"
            else:
                example_id, new_observations, new_title = prepare_round(df, all_annotate_rounds, total_rounds, user_id, next_round_id)


        res = [gr.HTML.update(new_instruction, visible=True), gr.HTML.update(new_title, visible=True),
               gr.HTML.update(new_observations, visible=True)]
        res += [gr.Number.update(label=question, value=0) for question in ages_questions]
        res += [gr.Radio.update(value=rating_scale[0]) for question, rating_scale in questions_ratings_filtered.items()]
        res += [next_button.update(new_button, visible=True)]
        res = tuple(res)
        return res


    # launch
    next_button.click(fn=annotation_round,
                      inputs=[name_box] + numbers + radios,
                      outputs=[instruction, title, observations] + numbers + radios + [next_button])
    block.queue(concurrency_count=1)
    block.launch(share=True)
