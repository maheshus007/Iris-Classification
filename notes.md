Create a Virtual Environment:

Navigate to your project folder (if you're not already in it):
cd /path/to/mlops-handson

Create a virtual environment:
python3 -m venv venv

Activate the Virtual Environment:
On macOS/Linux:
source venv/bin/activate

On Windows:
.\venv\Scripts\activate

Install Dependencies in the Virtual Environment:
Once the virtual environment is activated, install the dependencies from requirements.txt:
pip install -r requirements.txt

Deactivate the Virtual Environment (when done):

When you're finished working in the virtual environment, you can deactivate it:
deactivate

-------

uvicorn src.api:app --reload

-----

docker build -t fastapi-iris-classification .

docker compose up --build