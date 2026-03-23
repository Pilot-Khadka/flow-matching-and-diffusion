## **flow-matching-and-diffusion**
A minimal comparison of the two continuous-time generative models: flow matching and diffusion models.


### **installation**
First install **PyTorch** appropriate for your system.

To check your CUDA version:

```bash
nvidia-smi
```

#### for cuda 12.6:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### for cuda 12.8
```bash
uv pip install torch torchvision
```

Once pytorch is installed, run:

```bash
uv sync
```

Activate the virtual env with:
```bash
source .venv/bin/activate
```

### references:
* [MIT 6.S184: Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/2026/index.html)


