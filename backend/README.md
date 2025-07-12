cd backend
conda activate skin-diagnosis
CUDA_VISIBLE_DEVICES=-1 python main.py

cd backend/chatbot
conda activate skin-diagnosis
uvicorn sse_server:app --host 0.0.0.0 --port 8001 --reload

cd frontend
conda activate skin-diagnosis
npm start