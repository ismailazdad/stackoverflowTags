import itertools
import numpy
import joblib
import pandas as pd

from service.w2v_features import w2vFeatures

model_path = "modeles/"
multilabel_binarizer = joblib.load(model_path + "multilabel_binarizer.pkl", 'r')


class ShowResultS:

    def __init__(self, question=None):
        self.question = question

    # prediction with TDIDF
    # make vectorization and prediction
    def getTagTfIdfResult(self, cleaned_question, vector, model):
        X = vector.transform([cleaned_question])
        predict = model.predict(X)
        tags_predicts = multilabel_binarizer.inverse_transform(predict)
        tag_result = list(itertools.chain(*tags_predicts))
        return tag_result

    def getTagTfIdfResultWithProba(self, cleaned_question, vector, model):
        X = vector.transform([cleaned_question])
        predict = model.predict(X)
        predict_probas = model.predict_proba(X)
        tags_predicts = multilabel_binarizer.inverse_transform(predict)
        results = self.getResultsPredictionByWords(predict_probas,tags_predicts)
        return results

    # prediction with word2vec
    # make vectorization and prediction
    def getTagWord2vecResult(self, cleaned_question, vector, model):
        word = w2vFeatures()
        X = word.word2vec_features([cleaned_question], vector)
        predict = model.predict(X)
        tags_predicts = multilabel_binarizer.inverse_transform(predict)
        tag_result = list(itertools.chain(*tags_predicts))
        return tag_result


    # prediction with word2vec
    # make vectorization and prediction
    def getTagWord2vecResultWithProba(self, cleaned_question, vector, model):
        word = w2vFeatures()
        X = word.word2vec_features([cleaned_question], vector)
        predict = model.predict(X)
        predict_probas = model.predict_proba(X)
        tags_predicts = multilabel_binarizer.inverse_transform(predict)
        results = self.getResultsPredictionByWords(predict_probas,tags_predicts)
        return results

    def getResultsPredictionByWords(self,predict_proba,tags_predicts):
        df_predict_probas = pd.DataFrame(columns=['Tags', 'Probas'])
        df_predict_probas['Tags'] = multilabel_binarizer.classes_
        df_predict_probas['Probas'] = predict_proba.reshape(-1)
        # Select probas > 33%
        df_predict_probas = df_predict_probas[df_predict_probas['Probas'] >= 0.33] \
.sort_values('Probas', ascending=False)

        # Results
        results = {}
        results['Pred_Tags'] = tags_predicts
        results['Pred_Tags_Probab'] = df_predict_probas.set_index('Tags')['Probas'].to_dict()
        return results

if __name__ == '__init__':
    pass
