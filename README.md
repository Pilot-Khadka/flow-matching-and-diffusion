## **flow-matching-and-diffusion**

<table border="0"><tr>
<td width="60%" valign="top">

A minimal implementation of the two continuous-time generative models: **flow matching** and **diffusion models**.

</td>
<td width="40%" valign="middle">
<img src="assets/brownian_motion.gif" alt="Brownian motion trajectories converging to a Gaussian"/>
</td>
</tr></table>

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


