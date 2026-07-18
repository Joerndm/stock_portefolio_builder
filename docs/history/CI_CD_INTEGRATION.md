# CI/CD Test Integration Examples

## GitHub Actions

### Quick Tests (on every push)
```yaml
# .github/workflows/quick-tests.yml
name: Quick Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        pip install numpy pandas scikit-learn
    
    - name: Run quick tests
      run: |
        python test_reports/quick_test_runner.py
```

### Full Tests (on pull request to main)
```yaml
# .github/workflows/full-tests.yml
name: Full Test Suite

on:
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        pip install -r requirements_PY_3_12.txt
    
    - name: Run full test suite
      run: |
        python test_reports/master_test_suite.py --verbose
    
    - name: Check regression
      run: |
        python test_reports/master_test_suite.py --category regression
```

### Nightly Full Suite
```yaml
# .github/workflows/nightly.yml
name: Nightly Full Suite

on:
  schedule:
    - cron: '0 2 * * *'  # Run at 2 AM daily

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        pip install -r requirements_PY_3_12.txt
    
    - name: Run all tests
      run: |
        python test_reports/master_test_suite.py --verbose
    
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: test_reports/test_results/
```

## GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - quick-test
  - full-test
  - regression

quick_tests:
  stage: quick-test
  image: python:3.12
  before_script:
    - pip install numpy pandas scikit-learn
  script:
    - python test_reports/quick_test_runner.py
  only:
    - branches

full_tests:
  stage: full-test
  image: python:3.12
  before_script:
    - pip install -r requirements_PY_3_12.txt
  script:
    - python test_reports/master_test_suite.py --verbose
  only:
    - merge_requests
    - main

regression_tests:
  stage: regression
  image: python:3.12
  before_script:
    - pip install -r requirements_PY_3_12.txt
  script:
    - python test_reports/master_test_suite.py --category regression
  only:
    - merge_requests
  allow_failure: true
```

## Azure Pipelines

```yaml
# azure-pipelines.yml
trigger:
- main
- develop

pool:
  vmImage: 'ubuntu-latest'

stages:
- stage: QuickTests
  jobs:
  - job: QuickTest
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.12'
    
    - script: |
        pip install numpy pandas scikit-learn
        python test_reports/quick_test_runner.py
      displayName: 'Run Quick Tests'

- stage: FullTests
  condition: eq(variables['Build.Reason'], 'PullRequest')
  jobs:
  - job: FullTest
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.12'
    
    - script: |
        pip install -r requirements_PY_3_12.txt
        python test_reports/master_test_suite.py --verbose
      displayName: 'Run Full Test Suite'
```

## Local Pre-commit Hook

### Setup
```bash
# Copy this to .git/hooks/pre-commit and make executable
# chmod +x .git/hooks/pre-commit
```

### Pre-commit Hook Script
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running quick tests before commit..."

# Run quick tests
python test_reports/quick_test_runner.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Quick tests failed!"
    echo "Please fix the issues before committing."
    echo ""
    echo "To commit anyway (not recommended), use: git commit --no-verify"
    exit 1
fi

echo ""
echo "✅ All quick tests passed!"
exit 0
```

## Docker Integration

### Dockerfile for Testing
```dockerfile
# Dockerfile.test
FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install
COPY requirements_PY_3_12.txt .
RUN pip install --no-cache-dir -r requirements_PY_3_12.txt

# Copy test files
COPY test_reports/ test_reports/
COPY *.py .

# Run tests
CMD ["python", "test_reports/master_test_suite.py", "--verbose"]
```

### Docker Compose for Testing
```yaml
# docker-compose.test.yml
version: '3.8'

services:
  quick-test:
    build:
      context: .
      dockerfile: Dockerfile.test
    command: python test_reports/quick_test_runner.py
  
  full-test:
    build:
      context: .
      dockerfile: Dockerfile.test
    command: python test_reports/master_test_suite.py --verbose
```

### Run Tests in Docker
```bash
# Quick tests
docker-compose -f docker-compose.test.yml run quick-test

# Full tests
docker-compose -f docker-compose.test.yml run full-test
```

## Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    stages {
        stage('Quick Tests') {
            steps {
                sh 'pip install numpy pandas scikit-learn'
                sh 'python test_reports/quick_test_runner.py'
            }
        }
        
        stage('Full Tests') {
            when {
                branch 'main'
            }
            steps {
                sh 'pip install -r requirements_PY_3_12.txt'
                sh 'python test_reports/master_test_suite.py --verbose'
            }
        }
        
        stage('Regression Tests') {
            when {
                branch 'main'
            }
            steps {
                sh 'python test_reports/master_test_suite.py --category regression'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'test_reports/test_results/**/*', allowEmptyArchive: true
        }
        failure {
            mail to: 'team@example.com',
                 subject: "Test Failed: ${currentBuild.fullDisplayName}",
                 body: "Tests failed. Check ${env.BUILD_URL} for details."
        }
    }
}
```

## Make Targets

```makefile
# Makefile
.PHONY: test-quick test-full test-ml test-regression test-all

test-quick:
	@echo "Running quick tests..."
	python test_reports/quick_test_runner.py

test-functional:
	@echo "Running functional tests..."
	python test_reports/master_test_suite.py --category functional

test-ml:
	@echo "Running ML tests..."
	python test_reports/master_test_suite.py --category ml

test-integration:
	@echo "Running integration tests..."
	python test_reports/master_test_suite.py --category integration

test-regression:
	@echo "Running regression tests..."
	python test_reports/master_test_suite.py --category regression

test-performance:
	@echo "Running performance tests..."
	python test_reports/master_test_suite.py --category performance

test-all:
	@echo "Running full test suite..."
	python test_reports/master_test_suite.py --verbose

# Convenience aliases
test: test-quick
validate: test-full
```

### Usage
```bash
make test-quick      # Quick validation
make test-ml         # ML tests only
make test-all        # Full suite
make test            # Default (quick)
```

## Best Practices for CI/CD

### 1. Test Pyramid
```
         Full Suite (nightly)
              /\
             /  \
            /    \
           / Int  \
          /  Tests \
         /----------\
        /   ML Tests \
       /--------------\
      / Functional     \
     /    Tests         \
    /--------------------\
   /   Quick Tests (PR)   \
  /------------------------\
```

### 2. Recommended Strategy
- **On every push**: Quick tests (< 1 min)
- **On pull request**: Functional + ML tests (~5 min)
- **Before merge**: Full suite including integration (~10 min)
- **Nightly**: Full suite + regression + performance
- **Before release**: Complete validation with baseline update

### 3. Fail Fast
```yaml
# Run quick tests first, fail fast
- stage: quick
  script: python test_reports/quick_test_runner.py

# Only proceed if quick tests pass
- stage: full
  script: python test_reports/master_test_suite.py
  depends_on: quick
```

### 4. Parallel Execution
```yaml
# Run different categories in parallel
jobs:
  functional:
    script: python test_reports/master_test_suite.py --category functional
  
  ml:
    script: python test_reports/master_test_suite.py --category ml
  
  integration:
    script: python test_reports/master_test_suite.py --category integration
```

### 5. Caching
```yaml
# Cache dependencies for faster builds
- uses: actions/cache@v2
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements_PY_3_12.txt') }}
```

## Monitoring and Notifications

### Slack Notification Example
```python
# Add to master_test_suite.py
import requests

def notify_slack(success, summary):
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    if not webhook_url:
        return
    
    color = 'good' if success else 'danger'
    emoji = '✅' if success else '❌'
    
    message = {
        'attachments': [{
            'color': color,
            'title': f'{emoji} Test Suite Results',
            'text': summary,
            'footer': 'Stock Portfolio Builder',
            'ts': int(time.time())
        }]
    }
    
    requests.post(webhook_url, json=message)
```

### Email Notification
```python
# Add to master_test_suite.py
import smtplib
from email.mime.text import MIMEText

def send_email_notification(success, summary):
    if success:
        return  # Only notify on failure
    
    msg = MIMEText(summary)
    msg['Subject'] = '❌ Test Suite Failed'
    msg['From'] = 'tests@stockbuilder.com'
    msg['To'] = 'team@stockbuilder.com'
    
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()
```

## Summary

Choose the CI/CD integration that fits your workflow:

- **GitHub Actions**: Best for GitHub repositories
- **GitLab CI**: Best for GitLab repositories
- **Docker**: Platform-independent, reproducible
- **Pre-commit Hooks**: Catch issues before commit
- **Make**: Simple local automation

All approaches support the Master Test Suite's flexible execution model!
