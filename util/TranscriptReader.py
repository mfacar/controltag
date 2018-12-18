import glob
import pandas as pd
import re


class TranscriptReader:

    @staticmethod
    def transcripts_to_dataframe(directory):
        rows_list = []

        path = directory + '**/*TRANSCRIPT.csv'
        for filename in glob.iglob(path, recursive=True):
            transcript = pd.read_csv(filename, sep='\t')
            m = re.search("^\/(.+\/)*(\d+)_TRANSCRIPT.csv", filename)
            person_id = m.group(2)
            p = {}
            question = ""
            answer = ""
            lines = len(transcript)
            for i in range(0, lines):
                row = transcript.iloc[i]
                if (row["speaker"] == "Ellie") or (i == lines - 1):
                    p["personId"] = person_id
                    if "(" in str(question):
                        question = question[question.index("(") + 1:question.index(")")]
                    p["question"] = question
                    p["answer"] = answer
                    if question != "":
                        rows_list.append(p)
                    p = {}
                    answer = ""
                    question = row["value"]
                else:
                    answer = str(answer) + " " + str(row["value"])

        all_participants = pd.DataFrame(rows_list, columns=['personId', 'question', 'answer'])
        all_participants.to_csv(directory + 'all.csv', sep=',')
        print("File was created")
        return all_participants
