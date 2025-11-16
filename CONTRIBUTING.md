# Contributing to Smart Traffic Management System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Screenshots if applicable

### Suggesting Features

1. Check existing feature requests
2. Create a new issue with:
   - Use case description
   - Proposed solution
   - Alternative solutions considered
   - Impact assessment

### Development Workflow

1. **Fork the repository**

```bash
git clone https://github.com/your-username/Smart-Traffic-Management-System-using-ML-and-DSA.git
cd Smart-Traffic-Management-System-using-ML-and-DSA
```

2. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

3. **Make your changes**

- Write clean, readable code
- Follow existing code style
- Add tests for new features
- Update documentation

4. **Test your changes**

```bash
# Backend tests
cd backend
pytest tests/ -v

# ML service tests
cd ml-service
pytest tests/ -v

# Frontend tests
cd frontend
npm test
```

5. **Commit your changes**

```bash
git add .
git commit -m "feat: add traffic prediction caching"
```

**Commit Message Convention:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

6. **Push to your fork**

```bash
git push origin feature/your-feature-name
```

7. **Create Pull Request**

- Go to the original repository
- Click "New Pull Request"
- Select your fork and branch
- Fill in the PR template
- Request review from maintainers

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git

### Setup

```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# ML service setup
cd ml-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
```

### Running Locally

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or run services individually
make dev-backend   # Terminal 1
make dev-ml        # Terminal 2
make dev-frontend  # Terminal 3
```

## Code Style

### Python

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use Black for formatting
- Use Flake8 for linting

```bash
# Format code
black app/

# Lint code
flake8 app/ --max-line-length=100
```

### JavaScript/React

- Use ES6+ features
- Follow Airbnb style guide
- Use Prettier for formatting
- Use ESLint for linting

```bash
# Format code
npm run format

# Lint code
npm run lint
```

## Testing Guidelines

### Backend Tests

```python
# tests/test_routing.py
import pytest
from app.services.dsa.routing import RoutingService

def test_compute_route():
    # Arrange
    routing_service = RoutingService(graph)

    # Act
    result = routing_service.compute_route(...)

    # Assert
    assert result is not None
    assert len(result.path) > 0
```

### ML Service Tests

```python
# tests/test_lstm.py
import pytest
from models.lstm_model import TrafficPredictor

def test_prediction():
    predictor = TrafficPredictor()
    sequence = np.random.randn(12, 7)

    prediction = predictor.predict(sequence)

    assert isinstance(prediction, float)
    assert prediction >= 0
```

### Frontend Tests

```javascript
// src/components/__tests__/TrafficMap.test.js
import { render, screen } from '@testing-library/react';
import TrafficMap from '../TrafficMap';

test('renders map container', () => {
  render(<TrafficMap />);
  const mapElement = screen.getByTestId('traffic-map');
  expect(mapElement).toBeInTheDocument();
});
```

## Documentation

- Update README.md if needed
- Add docstrings to all functions
- Update API documentation
- Add comments for complex logic

```python
def compute_route(origin, destination, algorithm="a_star"):
    """
    Compute optimal route between two points.

    Args:
        origin (tuple): (latitude, longitude) of origin
        destination (tuple): (latitude, longitude) of destination
        algorithm (str): Algorithm to use ("a_star" or "dijkstra")

    Returns:
        RouteResult: Computed route with path and metrics

    Raises:
        ValueError: If coordinates are invalid
    """
    pass
```

## Pull Request Guidelines

### PR Title

Use conventional commit format:
```
feat: add emergency vehicle priority routing
fix: resolve database connection timeout
docs: update deployment guide
```

### PR Description

Include:
1. **Summary:** What does this PR do?
2. **Motivation:** Why is this change needed?
3. **Changes:** What was modified?
4. **Testing:** How was it tested?
5. **Screenshots:** If UI changes

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] No new warnings
- [ ] Backward compatible (or documented)

## Review Process

1. Automated checks must pass (CI/CD)
2. At least one maintainer approval required
3. Address review comments
4. Update PR if needed
5. Maintainer merges when approved

## Release Process

1. Update version in `pyproject.toml` and `package.json`
2. Update CHANGELOG.md
3. Create release branch: `release/v1.x.x`
4. Tag release: `git tag v1.x.x`
5. Push tag: `git push origin v1.x.x`
6. Create GitHub release with notes

## Community

- GitHub Discussions: Ask questions
- GitHub Issues: Report bugs
- Email: contribute@smarttraffic.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to reach out if you have questions!

Thank you for contributing! 🚀
