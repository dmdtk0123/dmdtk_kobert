from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json

from classify_text import get_classified_result

'''
def create_app():
    app = Flask(__name__)
    app.run(host='0.0.0.0', port='8000')
    CORS(app, resources={r"*": {"origins": "*"}})

    @app.route('/kobert-result', methods=["POST"])
    def kobert_result():
        if request.method == "POST":
            text_detail = request.form['productDetail']

            print("제품설명 DATA: " + text_detail)
            print(type(text_detail))

            classified_result_df = get_classified_result(text_detail)

        return classified_result_df.to_json(orient='records')

    if __name__ == '__main__':
        app.run('0.0.0.0',port=8000,debug=True)

    return app
'''

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/kobert-result', methods=["POST"])
def kobert_result():
    if request.method == "POST":
        text_detail = request.form['productDetail']
        text_notice = request.form['productNotice']
        text_price = request.form['productPrice']

        print("제품설명 DATA: " + text_detail)
        print(type(text_detail))

        classified_result_df = get_classified_result(text_detail)

        print()

        # notice_price = {'text_notice': text_notice,
        #                 'text_price': text_price}
        # notice_price = json.dumps(notice_price)

    # return classified_result_df.to_json(orient='records')
    #return classified_result_df.to_json(orient='index')
    classified_result_dict = json.loads(classified_result_df.to_json(orient='index'))
    classified_result_dict['text_notice'] = text_notice
    classified_result_dict['text_price'] = text_price

    return json.dumps(classified_result_dict)

if __name__ == '__main__':
    app.run(debug=True)

