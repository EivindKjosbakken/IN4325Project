#!/bin/bash

# Run npm and Python in the background and capture their PIDs
cd backend && python run.py &
python_pid=$!
cd frontend && npm run dev &
npm_pid=$!


# Wait for both processes to finish
wait $npm_pid
wait $python_pid

