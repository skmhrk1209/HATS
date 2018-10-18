import json
import requests
import subprocess


def read_lines(command):
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        shell=True
    ).stdout.readlines()


def post_slack(url, text):

    return requests.post(url, data=json.dumps({"text": text}))


if __name__ == "__main__":

    command = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"

    url = "https://hooks.slack.com/services/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"

    post_slack(url, "".join([line.decode("utf-8") + "\n" for line in read_lines(command)]))
