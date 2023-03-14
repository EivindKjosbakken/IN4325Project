#!/bin/bash

# Run npm and Python in the background and capture their PIDs
npm run dev &
npm_pid=$!
python run.py &
python_pid=$!

# Wait for both processes to finish
wait $npm_pid
wait $python_pid

