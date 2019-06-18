import sys
import os
from g2p_seq2seq import g2p
import g2p_seq2seq.g2p_trainer_utils as g2p_trainer_utils
from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.params import Params
from flask import Flask, jsonify

app = Flask(__name__)
params = Params("g2p-seq2seq", '')
params.hparams = g2p_trainer_utils.load_params("g2p-seq2seq")
model = G2PModel(params)

model.inputs = [] # initialization
model._G2PModel__prepare_interactive_model()

@app.route("/", methods=["GET"])
def index():
    output = model.decode_word(sys.argv[1])

    if (not output):
    	return "<h1></h1>"
    else:
        return "<h1>" + output[0] + "</h1>"
#         print(output[0]) # run for the first time


@app.route("/__health", methods=["GET"])
def health():
    """ Return a health response.

    The function can be written to always return a 200 response code,
    demonstrating that the application is running and able to respond, or it can
    perform a more substantive test for health.
    """

    def test():
        # A stub test for illustration only.
        try:
            os.geteuid()
        except OSError:
            return False
        else:
            return True

    if test():
        return "ok", 200
    else:
        return "not ok", 500


@app.route("/__stats/instance", methods=["GET"])
def instance_stats():
    mem_usage = ((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss +
                  resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) *
                 resource.getpagesize() / 1024.0)
    return jsonify({"MetricsKV": {"mem_kbs": mem_usage}, "MetricsVersion": 1})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)