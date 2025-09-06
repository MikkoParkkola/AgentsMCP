#!/bin/bash
set -euo pipefail

# AgentsMCP Infrastructure Validation Script
# Validates Docker build, Kubernetes manifests, and Helm chart

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FAILED_TESTS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

record_failure() {
    FAILED_TESTS+=("$1")
}

print_header() {
    echo "========================================"
    echo "$1"
    echo "========================================"
}

# Check if required tools are installed
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    local required_tools=("docker" "kubectl" "helm")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_success "$tool is installed"
        else
            log_error "$tool is not installed"
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and run again"
        exit 1
    fi
    
    # Check optional tools
    local optional_tools=("yamllint" "kubeval" "conftest")
    for tool in "${optional_tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_success "$tool is available (optional)"
        else
            log_warning "$tool is not installed (optional - enhances validation)"
        fi
    done
}

# Validate Dockerfile
validate_dockerfile() {
    print_header "Validating Dockerfile"
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -f "Dockerfile" ]]; then
        log_error "Dockerfile not found"
        record_failure "Dockerfile validation"
        return 1
    fi
    
    # Check Dockerfile syntax
    log_info "Checking Dockerfile syntax..."
    if docker build --dry-run -t agentsmcp:validate-test . >/dev/null 2>&1; then
        log_success "Dockerfile syntax is valid"
    else
        log_error "Dockerfile syntax validation failed"
        record_failure "Dockerfile syntax"
    fi
    
    # Security checks
    log_info "Checking Dockerfile security practices..."
    
    local security_issues=()
    
    # Check for non-root user
    if grep -q "USER.*1000" Dockerfile; then
        log_success "Non-root user configured (UID 1000)"
    else
        log_warning "Non-root user not explicitly set to UID 1000"
        security_issues+=("non-root user")
    fi
    
    # Check for multi-stage build
    if grep -q "FROM.*AS" Dockerfile; then
        log_success "Multi-stage build detected"
    else
        log_warning "Multi-stage build not detected"
        security_issues+=("multi-stage build")
    fi
    
    # Check for security labels
    if grep -q "LABEL.*security" Dockerfile; then
        log_success "Security labels found"
    else
        log_warning "Security labels not found"
    fi
    
    # Check for health check
    if grep -q "HEALTHCHECK" Dockerfile; then
        log_success "Health check configured"
    else
        log_warning "Health check not configured"
        security_issues+=("health check")
    fi
    
    if [[ ${#security_issues[@]} -gt 0 ]]; then
        log_warning "Security recommendations: ${security_issues[*]}"
    fi
}

# Build Docker image
build_docker_image() {
    print_header "Building Docker Image"
    
    cd "$PROJECT_ROOT"
    
    log_info "Building AgentsMCP Docker image..."
    
    if docker build -t agentsmcp:test .; then
        log_success "Docker image built successfully"
        
        # Inspect image for security
        log_info "Inspecting image security..."
        
        local image_user
        image_user=$(docker inspect --format='{{.Config.User}}' agentsmcp:test)
        if [[ "$image_user" == "1000" ]] || [[ "$image_user" == "appuser" ]]; then
            log_success "Image runs as non-root user: $image_user"
        else
            log_warning "Image user: $image_user (should be 1000 or appuser)"
        fi
        
        # Check image size
        local image_size
        image_size=$(docker images agentsmcp:test --format "{{.Size}}")
        log_info "Image size: $image_size"
        
        # Test container startup (quick test)
        log_info "Testing container startup..."
        if docker run --rm -d --name agentsmcp-test -p 8001:8000 agentsmcp:test >/dev/null 2>&1; then
            sleep 5
            if curl -f http://localhost:8001/health >/dev/null 2>&1; then
                log_success "Container starts and health check responds"
                docker stop agentsmcp-test >/dev/null 2>&1 || true
            else
                log_error "Health check failed"
                docker stop agentsmcp-test >/dev/null 2>&1 || true
                record_failure "Docker container health check"
            fi
        else
            log_error "Container failed to start"
            record_failure "Docker container startup"
        fi
    else
        log_error "Docker build failed"
        record_failure "Docker build"
    fi
}

# Validate Kubernetes manifests
validate_k8s_manifests() {
    print_header "Validating Kubernetes Manifests"
    
    local k8s_dir="$PROJECT_ROOT/k8s"
    
    if [[ ! -d "$k8s_dir" ]]; then
        log_error "k8s directory not found"
        record_failure "k8s directory"
        return 1
    fi
    
    # Basic YAML syntax validation
    log_info "Validating YAML syntax..."
    
    local yaml_files
    mapfile -t yaml_files < <(find "$k8s_dir" -name "*.yaml" -o -name "*.yml")
    
    for file in "${yaml_files[@]}"; do
        if command -v yamllint >/dev/null 2>&1; then
            if yamllint -d relaxed "$file" >/dev/null 2>&1; then
                log_success "YAML syntax valid: $(basename "$file")"
            else
                log_error "YAML syntax error in: $(basename "$file")"
                record_failure "YAML syntax: $(basename "$file")"
            fi
        else
            # Basic YAML check with kubectl
            if kubectl create --dry-run=client -f "$file" >/dev/null 2>&1; then
                log_success "Kubernetes YAML valid: $(basename "$file")"
            else
                log_error "Kubernetes YAML error in: $(basename "$file")"
                record_failure "Kubernetes YAML: $(basename "$file")"
            fi
        fi
    done
    
    # Validate with kubeval if available
    if command -v kubeval >/dev/null 2>&1; then
        log_info "Validating with kubeval..."
        if kubeval "$k8s_dir"/*.yaml >/dev/null 2>&1; then
            log_success "kubeval validation passed"
        else
            log_warning "kubeval validation had warnings"
        fi
    fi
    
    # Security policy validation with conftest if available
    if command -v conftest >/dev/null 2>&1; then
        log_info "Running security policy validation..."
        if conftest verify --policy "$PROJECT_ROOT/policies" "$k8s_dir"/*.yaml >/dev/null 2>&1; then
            log_success "Security policy validation passed"
        else
            log_warning "Security policy validation had issues (policies may not exist)"
        fi
    fi
    
    # Check for required resources
    local required_resources=("deployment" "service" "configmap" "secret" "rbac" "networkpolicy" "hpa")
    
    for resource in "${required_resources[@]}"; do
        if ls "$k8s_dir"/*"$resource"*.yaml >/dev/null 2>&1; then
            log_success "Found $resource manifest"
        else
            log_warning "$resource manifest not found"
        fi
    done
}

# Validate Helm chart
validate_helm_chart() {
    print_header "Validating Helm Chart"
    
    local chart_dir="$PROJECT_ROOT/charts/agentsmcp"
    
    if [[ ! -d "$chart_dir" ]]; then
        log_error "Helm chart directory not found"
        record_failure "Helm chart directory"
        return 1
    fi
    
    cd "$chart_dir"
    
    # Helm lint
    log_info "Running helm lint..."
    if helm lint . >/dev/null 2>&1; then
        log_success "Helm lint passed"
    else
        log_error "Helm lint failed"
        record_failure "Helm lint"
        helm lint .
    fi
    
    # Template rendering test
    log_info "Testing Helm template rendering..."
    if helm template agentsmcp-test . >/dev/null 2>&1; then
        log_success "Helm templates render successfully"
    else
        log_error "Helm template rendering failed"
        record_failure "Helm template rendering"
    fi
    
    # Test with different values files
    local values_files=("values-production.yaml" "values-staging.yaml" "values-development.yaml")
    
    for values_file in "${values_files[@]}"; do
        if [[ -f "$values_file" ]]; then
            log_info "Testing with $values_file..."
            if helm template agentsmcp-test . -f "$values_file" >/dev/null 2>&1; then
                log_success "Template rendering with $values_file successful"
            else
                log_error "Template rendering with $values_file failed"
                record_failure "Helm template: $values_file"
            fi
        else
            log_warning "$values_file not found"
        fi
    done
    
    # Validate required fields in Chart.yaml
    if [[ -f "Chart.yaml" ]]; then
        local required_fields=("name" "description" "version" "appVersion")
        
        for field in "${required_fields[@]}"; do
            if grep -q "^$field:" Chart.yaml; then
                log_success "Chart.yaml has required field: $field"
            else
                log_error "Chart.yaml missing required field: $field"
                record_failure "Chart.yaml: $field"
            fi
        done
    else
        log_error "Chart.yaml not found"
        record_failure "Chart.yaml missing"
    fi
}

# Security validation
validate_security() {
    print_header "Security Validation"
    
    # Check for hardcoded secrets
    log_info "Checking for hardcoded secrets..."
    
    local secret_patterns=("api_key" "password" "secret" "token" "credential")
    local found_secrets=()
    
    for pattern in "${secret_patterns[@]}"; do
        if grep -r -i "$pattern.*=.*[a-zA-Z0-9]" "$PROJECT_ROOT"/{k8s,charts} 2>/dev/null | grep -v "PLACEHOLDER" | grep -v "template" >/dev/null; then
            found_secrets+=("$pattern")
        fi
    done
    
    if [[ ${#found_secrets[@]} -eq 0 ]]; then
        log_success "No hardcoded secrets detected"
    else
        log_warning "Potential hardcoded secrets found: ${found_secrets[*]}"
        log_warning "Please ensure these are templates or placeholders"
    fi
    
    # Check security contexts
    log_info "Checking security contexts in manifests..."
    
    if grep -r "runAsNonRoot.*true" "$PROJECT_ROOT"/k8s/ >/dev/null 2>&1; then
        log_success "Non-root security context configured"
    else
        log_warning "Non-root security context not found"
    fi
    
    if grep -r "readOnlyRootFilesystem.*true" "$PROJECT_ROOT"/k8s/ >/dev/null 2>&1; then
        log_success "Read-only root filesystem configured"
    else
        log_warning "Read-only root filesystem not configured"
    fi
    
    if grep -r "allowPrivilegeEscalation.*false" "$PROJECT_ROOT"/k8s/ >/dev/null 2>&1; then
        log_success "Privilege escalation disabled"
    else
        log_warning "Privilege escalation not explicitly disabled"
    fi
}

# Resource validation
validate_resources() {
    print_header "Resource Configuration Validation"
    
    # Check resource limits and requests
    log_info "Checking resource limits and requests..."
    
    if grep -r "resources:" "$PROJECT_ROOT"/k8s/ >/dev/null 2>&1; then
        log_success "Resource configurations found"
        
        if grep -r "limits:" "$PROJECT_ROOT"/k8s/ >/dev/null 2>&1; then
            log_success "Resource limits configured"
        else
            log_warning "Resource limits not configured"
        fi
        
        if grep -r "requests:" "$PROJECT_ROOT"/k8s/ >/dev/null 2>&1; then
            log_success "Resource requests configured"
        else
            log_warning "Resource requests not configured"
        fi
    else
        log_warning "No resource configurations found"
    fi
    
    # Check health probes
    log_info "Checking health probes..."
    
    local probe_types=("livenessProbe" "readinessProbe" "startupProbe")
    
    for probe in "${probe_types[@]}"; do
        if grep -r "$probe:" "$PROJECT_ROOT"/k8s/ >/dev/null 2>&1; then
            log_success "$probe configured"
        else
            log_warning "$probe not configured"
        fi
    done
}

# Network policy validation
validate_network_policies() {
    print_header "Network Policy Validation"
    
    if [[ -f "$PROJECT_ROOT/k8s/networkpolicy.yaml" ]]; then
        log_success "Network policy manifest found"
        
        # Check for default deny policy
        if grep -q "podSelector: {}" "$PROJECT_ROOT/k8s/networkpolicy.yaml"; then
            log_success "Default deny policy configured"
        else
            log_warning "Default deny policy not found"
        fi
        
        # Check for ingress and egress rules
        if grep -q "policyTypes:" "$PROJECT_ROOT/k8s/networkpolicy.yaml"; then
            log_success "Policy types specified"
        else
            log_warning "Policy types not specified"
        fi
    else
        log_warning "Network policy manifest not found"
    fi
}

# Generate summary report
generate_summary() {
    print_header "Validation Summary"
    
    if [[ ${#FAILED_TESTS[@]} -eq 0 ]]; then
        log_success "All validations passed! 🎉"
        log_info "Your AgentsMCP infrastructure is ready for production deployment."
    else
        log_error "Some validations failed:"
        for failure in "${FAILED_TESTS[@]}"; do
            echo "  - $failure"
        done
        echo
        log_info "Please address the failed validations before deploying to production."
        exit 1
    fi
    
    echo
    log_info "Next steps:"
    echo "1. Build and push your Docker image to a registry"
    echo "2. Update image registry/tag in Helm values files"
    echo "3. Configure secrets for your environment"
    echo "4. Deploy using: helm install agentsmcp ./charts/agentsmcp -f values-<env>.yaml"
    echo "5. Verify deployment: kubectl get pods,svc,ingress -n <namespace>"
}

# Main execution
main() {
    log_info "Starting AgentsMCP Infrastructure Validation"
    echo
    
    check_prerequisites
    validate_dockerfile
    
    # Only build image if Docker validation passed
    if [[ ! " ${FAILED_TESTS[*]} " =~ " Dockerfile " ]]; then
        build_docker_image
    else
        log_warning "Skipping Docker build due to Dockerfile validation failures"
    fi
    
    validate_k8s_manifests
    validate_helm_chart
    validate_security
    validate_resources
    validate_network_policies
    
    generate_summary
}

# Run with error handling
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi