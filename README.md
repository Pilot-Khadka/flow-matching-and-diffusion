## **flow-matching-and-diffusion**

A minimal implementation of the two continuous-time generative models: **flow matching** and **diffusion models**.

<center>
<img src="assets/langevin.gif" alt="Brownian motion trajectories converging to a Gaussian"/>
</center>

### **installation**
First install **PyTorch** appropriate for your system. Check your CUDA version with `nvidia-smi`, then:

```bash
# for cuda 12.6
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# for cuda 12.8
uv pip install torch torchvision
```

Then install the rest of the dependencies:
```bash
uv sync && source .venv/bin/activate
```


### references:
* [MIT 6.S184: Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/2026/index.html)
* [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366)
* [Flow matching for generative modeling](https://arxiv.org/pdf/2210.02747)

