import json
import time
import requests
import subprocess

url = "https://hooks.slack.com/services/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"


def read_lines(command):
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        shell=True
    ).stdout.readlines()


def post_slack(url, text):

    return requests.post(url, data=json.dumps({"text": text}))


if __name__ == "__main__":

    start = time.time()

    while True:

        time.sleep(1800)

        post_slack(url, "".join([line.decode("utf-8") + "\n" for line in read_lines("nvidia-smi")]))
