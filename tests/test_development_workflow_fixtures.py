"""
Test fixtures and data generators for development workflow testing.

Provides realistic project structures, development scenarios, and test data.
"""

import json
import tempfile
import pytest
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentsmcp.models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus


# Project Structure Fixtures

@pytest.fixture
def microservice_project(tmp_path):
    """Create a realistic microservice project structure."""
    project_dir = tmp_path / "user_service"
    project_dir.mkdir()
    
    # Service structure
    (project_dir / "src").mkdir()
    (project_dir / "src" / "user_service").mkdir()
    (project_dir / "src" / "user_service" / "__init__.py").write_text("")
    
    # Models
    (project_dir / "src" / "user_service" / "models.py").write_text("""
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class User(BaseModel):
    id: Optional[int] = None
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
""")
    
    # API endpoints
    (project_dir / "src" / "user_service" / "api.py").write_text("""
from fastapi import FastAPI, HTTPException, Depends
from typing import List
from .models import User, UserCreate, UserUpdate
from .database import get_db
from .auth import get_current_user

app = FastAPI(title="User Service", version="1.0.0")

@app.post("/users/", response_model=User)
async def create_user(user: UserCreate, db=Depends(get_db)):
    # Implementation here
    pass

@app.get("/users/", response_model=List[User])
async def list_users(skip: int = 0, limit: int = 100, db=Depends(get_db)):
    # Implementation here
    pass

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int, db=Depends(get_db)):
    # Implementation here
    pass

@app.put("/users/{user_id}", response_model=User)
async def update_user(
    user_id: int, 
    user_update: UserUpdate, 
    db=Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Implementation here
    pass

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Implementation here
    pass
""")
    
    # Database layer
    (project_dir / "src" / "user_service" / "database.py").write_text("""
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./users.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
""")
    
    # Authentication
    (project_dir / "src" / "user_service" / "auth.py").write_text("""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from .models import User

security = HTTPBearer()

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

def get_current_user(token: str = Depends(security)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    # Implementation here
    pass
""")
    
    # Tests
    (project_dir / "tests").mkdir()
    (project_dir / "tests" / "__init__.py").write_text("")
    (project_dir / "tests" / "conftest.py").write_text("""
import pytest
from fastapi.testclient import TestClient
from src.user_service.api import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_user_data():
    return {
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpass123",
        "full_name": "Test User"
    }
""")
    
    (project_dir / "tests" / "test_models.py").write_text("""
import pytest
from src.user_service.models import User, UserCreate, UserUpdate

def test_user_model():
    user = User(
        id=1,
        email="test@example.com",
        username="testuser",
        full_name="Test User"
    )
    assert user.email == "test@example.com"
    assert user.is_active is True

def test_user_create_model():
    user_create = UserCreate(
        email="new@example.com",
        username="newuser",
        password="password123"
    )
    assert user_create.email == "new@example.com"
    assert user_create.username == "newuser"
""")
    
    (project_dir / "tests" / "test_api.py").write_text("""
import pytest
from fastapi.testclient import TestClient

def test_create_user(client, sample_user_data):
    response = client.post("/users/", json=sample_user_data)
    # Implementation would check actual response
    assert response.status_code in [200, 201, 422]  # Created or validation error

def test_list_users(client):
    response = client.get("/users/")
    # Implementation would check actual response
    assert response.status_code in [200, 401]  # OK or unauthorized

def test_get_user(client):
    response = client.get("/users/1")
    # Implementation would check actual response
    assert response.status_code in [200, 404, 401]  # OK, not found, or unauthorized
""")
    
    # Configuration files
    (project_dir / "requirements.txt").write_text("""
fastapi==0.104.1
uvicorn==0.24.0
pydantic[email]==2.4.2
sqlalchemy==2.0.23
python-jose[cryptography]==3.3.0
python-multipart==0.0.6
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.1
""")
    
    (project_dir / "pyproject.toml").write_text("""
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "user-service"
version = "1.0.0"
description = "User management microservice"
authors = [{name = "Development Team"}]
license = {text = "MIT"}
requires-python = ">=3.8"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
multi_line_output = 3
""")
    
    (project_dir / "Dockerfile").write_text("""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/

EXPOSE 8000

CMD ["uvicorn", "src.user_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
""")
    
    (project_dir / ".github").mkdir()
    (project_dir / ".github" / "workflows").mkdir()
    (project_dir / ".github" / "workflows" / "ci.yml").write_text("""
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --tb=short
    
    - name: Lint with black
      run: |
        black --check src/ tests/
    
    - name: Security scan
      run: |
        pip install bandit
        bandit -r src/
""")
    
    (project_dir / "README.md").write_text("""
# User Service

A microservice for user management with FastAPI.

## Features

- User registration and authentication
- CRUD operations for user management
- JWT-based authentication
- PostgreSQL database integration
- Comprehensive test suite
- Docker deployment ready

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the service:
   ```bash
   uvicorn src.user_service.api:app --reload
   ```

3. Run tests:
   ```bash
   pytest tests/ -v
   ```

## API Documentation

Once running, visit:
- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

This project follows standard Python development practices:
- Code formatting with Black
- Import sorting with isort
- Testing with pytest
- Type hints with Pydantic
""")
    
    return project_dir


@pytest.fixture
def frontend_project(tmp_path):
    """Create a realistic React frontend project structure."""
    project_dir = tmp_path / "user_frontend"
    project_dir.mkdir()
    
    # Source structure
    (project_dir / "src").mkdir()
    (project_dir / "src" / "components").mkdir()
    (project_dir / "src" / "pages").mkdir()
    (project_dir / "src" / "services").mkdir()
    (project_dir / "src" / "types").mkdir()
    
    # Components
    (project_dir / "src" / "components" / "UserForm.tsx").write_text("""
import React, { useState } from 'react';
import { User, UserCreate } from '../types/user';

interface UserFormProps {
  onSubmit: (user: UserCreate) => void;
  initialUser?: User;
  isEditing?: boolean;
}

export const UserForm: React.FC<UserFormProps> = ({
  onSubmit,
  initialUser,
  isEditing = false
}) => {
  const [formData, setFormData] = useState<UserCreate>({
    email: initialUser?.email || '',
    username: initialUser?.username || '',
    password: '',
    full_name: initialUser?.full_name || ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="email">Email</label>
        <input
          type="email"
          id="email"
          value={formData.email}
          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
          required
        />
      </div>
      
      <div>
        <label htmlFor="username">Username</label>
        <input
          type="text"
          id="username"
          value={formData.username}
          onChange={(e) => setFormData({ ...formData, username: e.target.value })}
          required
        />
      </div>
      
      <div>
        <label htmlFor="password">Password</label>
        <input
          type="password"
          id="password"
          value={formData.password}
          onChange={(e) => setFormData({ ...formData, password: e.target.value })}
          required={!isEditing}
        />
      </div>
      
      <button type="submit">
        {isEditing ? 'Update User' : 'Create User'}
      </button>
    </form>
  );
};
""")
    
    # Pages
    (project_dir / "src" / "pages" / "UserList.tsx").write_text("""
import React, { useEffect, useState } from 'react';
import { User } from '../types/user';
import { userService } from '../services/userService';

export const UserList: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const fetchedUsers = await userService.getUsers();
        setUsers(fetchedUsers);
      } catch (err) {
        setError('Failed to fetch users');
      } finally {
        setLoading(false);
      }
    };

    fetchUsers();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="user-list">
      <h2>Users</h2>
      <div className="grid gap-4">
        {users.map((user) => (
          <div key={user.id} className="border p-4 rounded">
            <h3>{user.full_name || user.username}</h3>
            <p>{user.email}</p>
            <p>Status: {user.is_active ? 'Active' : 'Inactive'}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
""")
    
    # Services
    (project_dir / "src" / "services" / "userService.ts").write_text("""
import { User, UserCreate, UserUpdate } from '../types/user';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class UserService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async getUsers(): Promise<User[]> {
    return this.request<User[]>('/users/');
  }

  async getUser(id: number): Promise<User> {
    return this.request<User>(`/users/${id}`);
  }

  async createUser(user: UserCreate): Promise<User> {
    return this.request<User>('/users/', {
      method: 'POST',
      body: JSON.stringify(user),
    });
  }

  async updateUser(id: number, user: UserUpdate): Promise<User> {
    return this.request<User>(`/users/${id}`, {
      method: 'PUT',
      body: JSON.stringify(user),
    });
  }

  async deleteUser(id: number): Promise<void> {
    await this.request(`/users/${id}`, {
      method: 'DELETE',
    });
  }
}

export const userService = new UserService();
""")
    
    # Types
    (project_dir / "src" / "types" / "user.ts").write_text("""
export interface User {
  id: number;
  email: string;
  username: string;
  full_name?: string;
  is_active: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface UserCreate {
  email: string;
  username: string;
  password: string;
  full_name?: string;
}

export interface UserUpdate {
  email?: string;
  username?: string;
  full_name?: string;
  is_active?: boolean;
}
""")
    
    # Tests
    (project_dir / "src" / "__tests__").mkdir()
    (project_dir / "src" / "__tests__" / "UserForm.test.tsx").write_text("""
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { UserForm } from '../components/UserForm';

describe('UserForm', () => {
  const mockOnSubmit = jest.fn();

  beforeEach(() => {
    mockOnSubmit.mockClear();
  });

  test('renders form fields', () => {
    render(<UserForm onSubmit={mockOnSubmit} />);
    
    expect(screen.getByLabelText('Email')).toBeInTheDocument();
    expect(screen.getByLabelText('Username')).toBeInTheDocument();
    expect(screen.getByLabelText('Password')).toBeInTheDocument();
  });

  test('submits form with correct data', () => {
    render(<UserForm onSubmit={mockOnSubmit} />);
    
    fireEvent.change(screen.getByLabelText('Email'), {
      target: { value: 'test@example.com' }
    });
    fireEvent.change(screen.getByLabelText('Username'), {
      target: { value: 'testuser' }
    });
    fireEvent.change(screen.getByLabelText('Password'), {
      target: { value: 'password123' }
    });
    
    fireEvent.click(screen.getByText('Create User'));
    
    expect(mockOnSubmit).toHaveBeenCalledWith({
      email: 'test@example.com',
      username: 'testuser',
      password: 'password123',
      full_name: ''
    });
  });
});
""")
    
    # Configuration files
    (project_dir / "package.json").write_text("""
{
  "name": "user-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "@types/node": "^16.18.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "typescript": "^4.9.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "lint": "eslint src --ext .ts,.tsx",
    "format": "prettier --write src/**/*.{ts,tsx}"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^5.16.0",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^14.4.0",
    "@types/jest": "^27.5.0",
    "@typescript-eslint/eslint-plugin": "^5.62.0",
    "@typescript-eslint/parser": "^5.62.0",
    "eslint": "^8.45.0",
    "prettier": "^2.8.0"
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
""")
    
    (project_dir / "tsconfig.json").write_text("""
{
  "compilerOptions": {
    "target": "es5",
    "lib": [
      "dom",
      "dom.iterable",
      "es6"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": [
    "src"
  ]
}
""")
    
    return project_dir


# Development Scenario Fixtures

@pytest.fixture
def development_scenarios():
    """Provide realistic development scenarios for testing."""
    return [
        {
            "name": "user_authentication_feature",
            "description": "Implement user authentication with JWT tokens",
            "phases": [
                {
                    "name": "requirements_gathering",
                    "roles": ["business_analyst"],
                    "deliverables": ["requirements_document", "user_stories", "acceptance_criteria"]
                },
                {
                    "name": "backend_implementation",
                    "roles": ["backend_engineer", "api_engineer"],
                    "deliverables": ["authentication_service", "jwt_middleware", "user_endpoints"]
                },
                {
                    "name": "frontend_implementation", 
                    "roles": ["web_frontend_engineer"],
                    "deliverables": ["login_form", "register_form", "auth_context"]
                },
                {
                    "name": "testing",
                    "roles": ["backend_qa_engineer", "web_frontend_qa_engineer"],
                    "deliverables": ["unit_tests", "integration_tests", "e2e_tests"]
                },
                {
                    "name": "deployment",
                    "roles": ["ci_cd_engineer"],
                    "deliverables": ["ci_pipeline", "deployment_scripts", "monitoring"]
                }
            ],
            "estimated_duration": "2 weeks",
            "complexity": "medium"
        },
        {
            "name": "user_profile_management",
            "description": "Allow users to manage their profiles",
            "phases": [
                {
                    "name": "analysis",
                    "roles": ["business_analyst"],
                    "deliverables": ["profile_requirements", "privacy_analysis"]
                },
                {
                    "name": "backend_development",
                    "roles": ["backend_engineer"],
                    "deliverables": ["profile_service", "validation_rules"]
                },
                {
                    "name": "frontend_development",
                    "roles": ["web_frontend_engineer", "tui_frontend_engineer"],
                    "deliverables": ["profile_forms", "settings_pages"]
                },
                {
                    "name": "quality_assurance",
                    "roles": ["backend_qa_engineer", "web_frontend_qa_engineer"],
                    "deliverables": ["test_suites", "security_tests"]
                }
            ],
            "estimated_duration": "1 week",
            "complexity": "low"
        },
        {
            "name": "real_time_notifications",
            "description": "Implement real-time push notifications",
            "phases": [
                {
                    "name": "architecture_design",
                    "roles": ["backend_engineer", "api_engineer"],
                    "deliverables": ["architecture_document", "technology_choices"]
                },
                {
                    "name": "backend_infrastructure",
                    "roles": ["backend_engineer", "ci_cd_engineer"],
                    "deliverables": ["websocket_service", "message_queue", "notification_service"]
                },
                {
                    "name": "client_integration",
                    "roles": ["web_frontend_engineer", "tui_frontend_engineer"],
                    "deliverables": ["websocket_client", "notification_ui", "offline_handling"]
                },
                {
                    "name": "comprehensive_testing",
                    "roles": ["backend_qa_engineer", "web_frontend_qa_engineer", "chief_qa_engineer"],
                    "deliverables": ["load_tests", "reliability_tests", "cross_platform_tests"]
                }
            ],
            "estimated_duration": "3 weeks",
            "complexity": "high"
        }
    ]


@pytest.fixture
def task_envelope_factory():
    """Factory for creating TaskEnvelopeV1 instances."""
    def create_task(
        objective: str,
        module: Optional[str] = None,
        write_paths: Optional[List[str]] = None,
        read_only_paths: Optional[List[str]] = None,
        time_limit: int = 300,
        cost_limit: float = 1.0
    ) -> TaskEnvelopeV1:
        return TaskEnvelopeV1(
            objective=objective,
            bounded_context=f"test_project/{module or 'default'}",
            inputs={},
            constraints=[
                f"time_limit: {time_limit}s",
                f"cost_limit: {cost_limit}eur",
                f"write_paths: {','.join(write_paths or [])}",
                f"read_only_paths: {','.join(read_only_paths or [])}"
            ]
        )
    
    return create_task


@pytest.fixture
def result_envelope_factory():
    """Factory for creating ResultEnvelopeV1 instances."""
    def create_result(
        status: EnvelopeStatus = EnvelopeStatus.SUCCESS,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        notes: str = "Task completed successfully",
        confidence: float = 0.9
    ) -> ResultEnvelopeV1:
        return ResultEnvelopeV1(
            status=status,
            artifacts=artifacts or [],
            notes=notes,
            confidence=confidence,
            metrics={
                "lat_ms": 1500,
                "tok_in": 500,
                "tok_out": 200,
                "eur": 0.05
            }
        )
    
    return create_result


# Agent Response Fixtures

@pytest.fixture
def realistic_agent_responses():
    """Provide realistic agent responses for different roles."""
    return {
        "business_analyst": {
            "user_authentication": {
                "requirements": [
                    "Users must be able to register with email and password",
                    "Users must be able to login with credentials",
                    "System must support password reset functionality",
                    "Session management with secure tokens",
                    "Account lockout after failed attempts"
                ],
                "user_stories": [
                    "As a new user, I want to register an account so I can access the platform",
                    "As a returning user, I want to login quickly so I can continue my work",
                    "As a user, I want to reset my password if I forget it"
                ],
                "acceptance_criteria": [
                    "Registration form validates email format and password strength",
                    "Login returns JWT token valid for 24 hours", 
                    "Password reset sends email with secure token",
                    "Account locks after 5 failed login attempts"
                ],
                "non_functional_requirements": [
                    "Login response time < 500ms",
                    "Support 10,000 concurrent users",
                    "99.9% availability for authentication service"
                ]
            }
        },
        "backend_engineer": {
            "user_authentication": {
                "architecture": {
                    "database_schema": {
                        "users": ["id", "email", "username", "password_hash", "created_at"],
                        "sessions": ["id", "user_id", "token", "expires_at", "created_at"]
                    },
                    "services": [
                        "AuthenticationService",
                        "UserService", 
                        "TokenService",
                        "PasswordService"
                    ],
                    "security": {
                        "password_hashing": "bcrypt with salt",
                        "jwt_algorithm": "HS256",
                        "token_expiry": "24 hours",
                        "refresh_token": "30 days"
                    }
                },
                "api_endpoints": [
                    "POST /auth/register",
                    "POST /auth/login", 
                    "POST /auth/refresh",
                    "POST /auth/logout",
                    "POST /auth/password-reset"
                ],
                "implementation_tasks": [
                    "Setup database migrations",
                    "Implement password hashing utilities",
                    "Create JWT token management",
                    "Build authentication middleware",
                    "Add input validation and sanitization"
                ]
            }
        },
        "web_frontend_engineer": {
            "user_authentication": {
                "components": [
                    "LoginForm",
                    "RegisterForm",
                    "PasswordResetForm",
                    "AuthProvider",
                    "ProtectedRoute"
                ],
                "pages": [
                    "LoginPage",
                    "RegisterPage", 
                    "ForgotPasswordPage",
                    "Dashboard"
                ],
                "state_management": {
                    "auth_context": "User authentication state",
                    "token_storage": "Secure local storage",
                    "auto_logout": "Token expiry handling"
                },
                "ui_ux": {
                    "design_system": "Material-UI components",
                    "responsive": "Mobile-first design",
                    "accessibility": "WCAG 2.1 AA compliance",
                    "loading_states": "Skeleton screens and spinners"
                }
            }
        },
        "backend_qa_engineer": {
            "user_authentication": {
                "test_strategy": {
                    "unit_tests": "Service layer and utilities",
                    "integration_tests": "API endpoints with database",
                    "contract_tests": "API specifications",
                    "security_tests": "Authentication flows and vulnerabilities"
                },
                "test_cases": [
                    "Valid user registration",
                    "Duplicate email registration",
                    "Invalid password format",
                    "Successful login",
                    "Invalid credentials",
                    "Account lockout",
                    "Password reset flow",
                    "Token expiry handling"
                ],
                "performance_tests": [
                    "Login endpoint load test",
                    "Concurrent registration test",
                    "Database connection pooling",
                    "Memory usage under load"
                ],
                "security_tests": [
                    "SQL injection attempts",
                    "Cross-site scripting (XSS)",
                    "Brute force attack protection",
                    "JWT token validation",
                    "Password strength enforcement"
                ]
            }
        }
    }


# Development Metrics Fixtures

@pytest.fixture
def development_metrics():
    """Provide realistic development metrics for testing."""
    return {
        "velocity": {
            "story_points_per_sprint": 25,
            "completed_stories": 8,
            "team_capacity": 40,
            "velocity_trend": "stable"
        },
        "quality": {
            "code_coverage": 0.85,
            "defect_density": 0.02,
            "technical_debt_ratio": 0.15,
            "maintainability_index": 8.5
        },
        "performance": {
            "build_time": 180,  # seconds
            "test_execution_time": 45,  # seconds
            "deployment_frequency": 3,  # per week
            "lead_time": 2.5  # days
        },
        "team_productivity": {
            "commits_per_day": 12,
            "pull_requests_per_week": 8,
            "code_review_time": 4,  # hours
            "time_to_merge": 1.5  # days
        }
    }


if __name__ == "__main__":
    # Test the fixtures
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Test microservice project creation
        class MockRequest:
            class Node:
                name = "test_microservice"
            node = Node()
        
        # This would normally be called by pytest
        project = microservice_project.__wrapped__(tmp_path)
        print(f"Created microservice project at: {project}")
        print(f"Project has {len(list(project.rglob('*')))} files/directories")
        
        # Test task envelope factory
        create_task = task_envelope_factory.__wrapped__()
        task = create_task("Implement user authentication", "auth", ["src/auth/"])
        print(f"Created task: {task.objective}")
        print(f"Task constraints: {task.constraints}")