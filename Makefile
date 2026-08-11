build:
	pip install -e .

test:
	pytest tests/ -v

benchmark:
	python evals/benchmark.py

profile:
	mkdir -p profiles && ncu --set full -o profiles/voxelize_profile \
		python benchmarks/profile_kernel.py

clean:
	rm -rf build/ *.egg-info voxelize_cuda*.so __pycache__/
