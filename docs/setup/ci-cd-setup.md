# 📋 QLORAX CI/CD Setup Instructions

This document provides step-by-step instructions to configure your QLORAX project for the automated CI/CD pipeline.

## 🔧 Prerequisites

Before setting up the CI/CD pipeline, ensure you have:

1. **GitHub Repository:** Your QLORAX project hosted on GitHub
2. **Docker Hub Account:** For container registry (or use GitHub Container Registry)
3. **Hugging Face Account:** For model publishing
4. **Weights & Biases Account:** For experiment tracking (optional)

## ⚙️ GitHub Repository Setup

### 1. Repository Secrets

Navigate to your GitHub repository → Settings → Secrets and Variables → Actions, and add these secrets:

#### Required Secrets:
```
HF_TOKEN=your_huggingface_api_token
DOCKER_USERNAME=your_docker_hub_username  
DOCKER_PASSWORD=your_docker_hub_password_or_access_token
```

#### Optional Secrets:
```
WANDB_API_KEY=your_wandb_api_key
HF_ORGANIZATION=your_hf_organization_name
```

### 2. Repository Variables

Add these repository variables for configuration:

```
PYTHON_VERSION=3.11
DEFAULT_DOMAIN=artificial_intelligence
DEFAULT_SAMPLES=25
```

## 🐳 Docker Setup

### 1. Create Docker Hub Repository

1. Log in to [Docker Hub](https://hub.docker.com)
2. Create repository: `your-username/qlorax-serve`
3. Set visibility to Public or Private as needed

### 2. Test Docker Build Locally

```bash
# Test training container
docker build -f Dockerfile.training -t qlorax-trainer:test .

# Test serving container  
docker build -f Dockerfile.serve -t qlorax-serve:test .

# Test with docker-compose
docker-compose --profile serving up -d
```

## 🤗 Hugging Face Setup

### 1. Create Access Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create new token with **Write** permissions
3. Add to GitHub secrets as `HF_TOKEN`

### 2. Create Organization (Optional)

1. Create organization on Hugging Face
2. Add organization name to GitHub secrets as `HF_ORGANIZATION`

## 🚀 CI/CD Pipeline Activation

### 1. Enable GitHub Actions

The pipeline is automatically triggered by:
- **Push to main/develop branches**
- **Pull requests to main branch**
- **Manual workflow dispatch**

### 2. First Pipeline Run

1. Make a commit to trigger the pipeline:
```bash
git add .
git commit -m "feat: activate QLORAX CI/CD pipeline"
git push origin main
```

2. Monitor progress in GitHub Actions tab

### 3. Manual Pipeline Trigger

1. Go to GitHub Actions → QLORAX QLoRA Fine-Tuning Pipeline
2. Click "Run workflow"
3. Configure parameters:
   - Domain: artificial_intelligence, machine_learning, etc.
   - Number of samples: 25 (default)
   - Force training: false (default)

## 📊 Quality Gates Configuration

### Default Thresholds

The pipeline uses these quality gates:

```json
{
  "thresholds": {
    "bert_f1_minimum": 0.90,
    "rouge_l_minimum": 0.85,
    "coherence_minimum": 0.85,
    "response_time_maximum": 2.0,
    "model_size_maximum_mb": 50,
    "training_loss_maximum": 2.0
  }
}
```

### Customizing Quality Gates

1. Edit `configs/quality-gates.json`
2. Adjust thresholds based on your requirements
3. Commit changes to trigger pipeline update

## 🔍 Monitoring and Debugging

### 1. Pipeline Status

Monitor pipeline progress:
- GitHub Actions tab shows real-time status
- Each job provides detailed logs
- Artifacts are automatically saved

### 2. Common Issues

#### Docker Build Failures
```bash
# Test locally first
docker build -f Dockerfile.training -t test .

# Check Docker Hub credentials
docker login
```

#### Quality Gate Failures
```bash
# Review evaluation results
cat results/quality-gate.json

# Adjust thresholds if needed
edit configs/quality-gates.json
```

#### HuggingFace Upload Issues
```bash
# Test token permissions
python -c "from huggingface_hub import HfApi; api = HfApi(token='your_token'); print(api.whoami())"

# Check organization access
python scripts/huggingface_publisher.py list
```

### 3. Pipeline Artifacts

Each successful run generates:
- **Trained Model:** LoRA adapters and configuration
- **Evaluation Results:** Comprehensive benchmarking data
- **Quality Gate Reports:** Pass/fail status with metrics
- **Docker Images:** Ready-to-deploy containers

## 🎯 Production Deployment

### 1. Automatic Deployment

After successful pipeline completion:
- Model is published to Hugging Face Hub
- Docker container is pushed to registry
- Both are ready for production use

### 2. Manual Deployment

Using Docker:
```bash
# Pull latest serving container
docker pull your-username/qlorax-serve:latest

# Run with environment variables
docker run -d \
  -p 7860:7860 \
  -p 8000:8000 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  your-username/qlorax-serve:latest
```

Using Hugging Face:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load your published model
model = PeftModel.from_pretrained(
    base_model, 
    "your-org/qlorax-enhanced-model-abc123"
)
```

### 3. Production Scaling

For production environments:
```bash
# Use production docker-compose profile
docker-compose --profile production up -d

# This includes:
# - Nginx reverse proxy
# - PostgreSQL database  
# - Redis caching
# - Prometheus monitoring
# - Grafana dashboards
```

## 🔧 Advanced Configuration

### 1. Multi-Environment Setup

Create branch-specific configurations:
- `main` branch → Production deployment
- `develop` branch → Staging environment
- Feature branches → Development testing

### 2. Custom Triggers

Modify `.github/workflows/qlorax-cicd.yml` for custom triggers:
```yaml
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly training on Sundays
  workflow_call:
    inputs:
      custom_config:
        required: false
        type: string
```

### 3. Integration with External Systems

Add webhook notifications:
```yaml
- name: Notify Slack
  if: always()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 📈 Performance Optimization

### 1. Caching Strategy

The pipeline uses caching for:
- Python dependencies (`actions/setup-python@v4` with cache)
- Docker layer caching (`docker/build-push-action@v5`)
- Model artifacts (GitHub Actions artifacts)

### 2. Parallel Execution

Jobs run in parallel where possible:
- Data preparation runs first
- Dry run and training can run simultaneously (with conditions)
- Evaluation depends on training completion
- Publishing runs in parallel after evaluation

### 3. Resource Management

Configure runners based on workload:
- **CPU-intensive jobs:** Use larger GitHub runners
- **GPU training:** Use self-hosted runners with GPU
- **Memory requirements:** Adjust Docker resource limits

## 🎉 Success Indicators

A successful CI/CD setup will show:

✅ **Automated Triggers:** Pipeline runs on every push/PR  
✅ **Quality Gates:** Consistent passing of evaluation thresholds  
✅ **Artifact Publishing:** Models automatically published to HF Hub  
✅ **Container Deployment:** Docker images ready for production  
✅ **Monitoring Integration:** Real-time metrics and alerting  

## 📞 Support and Troubleshooting

If you encounter issues:

1. **Check GitHub Actions logs** for detailed error messages
2. **Review quality gate results** in artifacts
3. **Test components locally** before pushing
4. **Validate secrets and tokens** are correctly configured
5. **Monitor resource usage** and adjust limits as needed

For additional support, refer to:
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)