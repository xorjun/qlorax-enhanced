# Contributing to QLORAX Enhanced

Thank you for your interest in contributing to QLORAX Enhanced! 🎉 This document provides guidelines and information for contributors.

## 🎯 **How to Contribute**

### **Types of Contributions**

We welcome several types of contributions:

- 🐛 **Bug reports and fixes**
- ✨ **New features and enhancements**
- 📚 **Documentation improvements**
- 🧪 **Test coverage expansion**
- 🔧 **Performance optimizations**
- 🌐 **UI/UX improvements**
- 🔬 **InstructLab enhancements**

## 🚀 **Getting Started**

### **1. Fork and Clone**

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/qlorax-enhanced.git
cd qlorax-enhanced

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/qlorax-enhanced.git
```

### **2. Set Up Development Environment**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-instructlab.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### **3. Create Development Branch**

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

## 📋 **Development Guidelines**

### **Code Style**

We use automated code formatting and linting:

```bash
# Format code
black .
isort .

# Check linting
flake8 scripts/ --max-line-length=88

# Type checking
mypy scripts/ --ignore-missing-imports
```

### **Code Quality Standards**

- ✅ **Type hints** for function signatures
- ✅ **Docstrings** for classes and functions  
- ✅ **Error handling** with informative messages
- ✅ **Logging** instead of print statements
- ✅ **Unit tests** for new functionality
- ✅ **Performance considerations**

### **Example Code Style**

```python
#!/usr/bin/env python3
"""
Module for enhanced training functionality.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedTrainer:
    """Enhanced training class with InstructLab integration.
    
    Args:
        config_path: Path to training configuration file
        use_instructlab: Whether to enable InstructLab features
        
    Raises:
        FileNotFoundError: If configuration file is not found
        ValueError: If configuration is invalid
    """
    
    def __init__(self, config_path: str, use_instructlab: bool = True) -> None:
        self.config_path = Path(config_path)
        self.use_instructlab = use_instructlab
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        logger.info(f"Initialized trainer with config: {config_path}")
    
    def train(self, epochs: int = 3) -> Dict[str, Any]:
        """Execute training pipeline.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Training results with metrics and model path
            
        Raises:
            RuntimeError: If training fails
        """
        try:
            logger.info(f"Starting training for {epochs} epochs")
            # Training implementation here
            
            results = {
                "status": "success",
                "epochs": epochs,
                "metrics": {"loss": 0.1, "accuracy": 0.95}
            }
            
            logger.info("Training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training execution failed: {e}") from e
```

## 🧪 **Testing Guidelines**

### **Writing Tests**

```bash
# Create test file: tests/test_your_feature.py
import pytest
from scripts.your_module import YourClass

def test_your_feature():
    """Test your feature functionality."""
    # Arrange
    instance = YourClass(config="test-config.yaml")
    
    # Act
    result = instance.your_method()
    
    # Assert
    assert result["status"] == "success"
    assert result["value"] > 0
```

### **Running Tests**

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_your_feature.py -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=term-missing
```

### **Integration Tests**

```bash
# Test InstructLab integration
python test_integration.py

# Test system validation
python validate_system.py
```

## 📊 **Performance Guidelines**

### **Benchmarking**

Always benchmark performance-critical changes:

```bash
# Run enhanced benchmarks
python scripts/enhanced_benchmark.py \
    --model models/demo-qlora-model \
    --test-data data/test_data.jsonl

# Check quality gates
python scripts/quality_gates.py --stage evaluation
```

### **Performance Expectations**

- 🎯 **BERT F1**: ≥ 90%
- 🎯 **ROUGE-L**: ≥ 85%
- 🎯 **BLEU**: ≥ 80%
- ⚡ **Training Time**: Should not increase by >20%
- 💾 **Memory Usage**: Monitor for memory leaks

## 📚 **Documentation Guidelines**

### **Code Documentation**

- Use clear, descriptive docstrings
- Include parameter types and descriptions
- Document return values and exceptions
- Add usage examples for complex functions

### **User Documentation**

```bash
# Update relevant documentation
docs/guides/your-feature-guide.md
docs/reference/your-feature-reference.md

# Update main README if needed
README.md
```

## 🔄 **Pull Request Process**

### **1. Pre-submission Checklist**

- [ ] ✅ Code follows style guidelines
- [ ] 🧪 Tests added/updated and passing
- [ ] 📚 Documentation updated
- [ ] 🔍 Code reviewed locally
- [ ] 📊 Performance benchmarks run
- [ ] 🚨 No new security issues

### **2. Commit Messages**

Use conventional commit format:

```bash
# Feature addition
git commit -m "✨ feat: add InstructLab batch processing support

- Implement batch synthetic data generation
- Add progress tracking and error handling
- Include comprehensive test coverage
- Update documentation with usage examples

Resolves #123"

# Bug fix
git commit -m "🐛 fix: resolve memory leak in training loop

- Fix tensor cleanup in validation phase
- Add proper garbage collection
- Update error handling for CUDA OOM
- Add regression test

Fixes #456"

# Documentation
git commit -m "📚 docs: update InstructLab integration guide

- Add troubleshooting section
- Include performance optimization tips
- Update code examples
- Fix broken links"
```

### **3. Pull Request Template**

```markdown
## 📋 **Description**

Brief description of changes and motivation.

## 🔄 **Changes Made**

- [ ] Feature 1
- [ ] Feature 2
- [ ] Bug fix

## 🧪 **Testing**

- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Performance benchmarks run
- [ ] Manual testing completed

## 📊 **Performance Impact**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| BERT F1 | 95.2% | 95.8% | +0.6% |
| Training Time | 2.5h | 2.3h | -8% |

## 📚 **Documentation**

- [ ] Code comments updated
- [ ] User documentation updated
- [ ] API documentation updated

## ✅ **Checklist**

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Performance benchmarks acceptable
```

### **4. Review Process**

1. **Automated Checks**: GitHub Actions will run all tests
2. **Code Review**: Maintainers will review your changes
3. **Performance Check**: Benchmarks must meet thresholds
4. **Documentation Review**: Ensure docs are updated
5. **Final Approval**: Maintainer approval required

## 🐛 **Reporting Issues**

### **Bug Reports**

Use the bug report template:

```markdown
**🐛 Bug Description**
Clear description of the bug

**🔄 Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**✅ Expected Behavior**
What should happen

**❌ Actual Behavior**
What actually happens

**🖥️ Environment**
- OS: [e.g. Ubuntu 20.04]
- Python: [e.g. 3.11.0]
- QLORAX Version: [e.g. v2.1.0]

**📊 Additional Context**
Error messages, logs, screenshots
```

### **Feature Requests**

```markdown
**✨ Feature Description**
Clear description of the proposed feature

**🎯 Use Case**
Why is this feature needed?

**💡 Proposed Solution**
How should this work?

**🔄 Alternatives Considered**
Other approaches you considered

**📊 Additional Context**
Mockups, examples, references
```

## 🏷️ **Release Process**

### **Versioning**

We use [Semantic Versioning](https://semver.org/):

- **MAJOR** (v2.0.0): Breaking changes
- **MINOR** (v2.1.0): New features, backward compatible
- **PATCH** (v2.1.1): Bug fixes, backward compatible

### **Release Workflow**

1. **Feature Freeze**: Stop adding new features
2. **Release Branch**: Create `release/v2.1.0` branch
3. **Testing**: Comprehensive testing and validation
4. **Documentation**: Update changelogs and docs
5. **Tag Release**: Create git tag `v2.1.0`
6. **Deploy**: Automated deployment via GitHub Actions

## 🤝 **Community**

### **Getting Help**

- 📚 **Documentation**: Check docs/ folder first
- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Report bugs via GitHub Issues
- 📧 **Direct Contact**: For security issues only

### **Code of Conduct**

We follow the [Contributor Covenant](https://www.contributor-covenant.org/):

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Maintain a positive environment

## 🙏 **Recognition**

Contributors are recognized in:

- 📚 **README.md** acknowledgments section
- 🏷️ **Release notes** for significant contributions
- 📊 **GitHub contributors** page
- 🎉 **Special mentions** in project updates

## 📊 **Development Resources**

### **Useful Commands**

```bash
# Full development setup
make dev-setup  # (if Makefile exists)

# Run all checks
make check      # Format, lint, type-check, test

# Performance benchmark
make benchmark

# Documentation build
make docs
```

### **IDE Setup**

**VS Code** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true
}
```

## 🎯 **Priority Areas**

Current priority areas for contributions:

1. **🔬 InstructLab Enhancement**
   - Better synthetic data quality
   - More domain taxonomies
   - Performance optimization

2. **📊 Evaluation Metrics**
   - New evaluation methods
   - Benchmark improvements
   - Quality gate enhancements

3. **🌐 User Interface**
   - Gradio UI improvements
   - Better visualizations
   - Mobile responsiveness

4. **🐳 Deployment**
   - Kubernetes support
   - Cloud platform integration
   - Monitoring improvements

---

Thank you for contributing to QLORAX Enhanced! 🎉

*Together we're building the best QLoRA fine-tuning suite available!*