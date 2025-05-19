from main import app, db
from ml import MachineLearning
import traceback
#python test_ml.py
'''if __name__ == '__main__':
    try:
        ml = MachineLearning(db, app)

        test_user_id = 1
        #ieraksti kas saistīti ar medicīnu:
        liked_ids = [30, 22, 20]

        with app.app_context():
            print("Training model...")
            ml.train(test_user_id, liked_ids)

           # sample_text = "Medicīnas mēbeļu un aprīkojuma iegāde Sterilizācijas nodaļai"
            sample_text = "RTU ēku un inženiersistēmu apkalpošana un remonts"
           
           
            prediction = ml.predict(test_user_id, sample_text)

            print(prediction)

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()'''
