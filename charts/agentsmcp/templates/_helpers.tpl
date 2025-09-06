{{/*
Expand the name of the chart.
*/}}
{{- define "agentsmcp.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "agentsmcp.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "agentsmcp.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "agentsmcp.labels" -}}
helm.sh/chart: {{ include "agentsmcp.chart" . }}
{{ include "agentsmcp.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "agentsmcp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentsmcp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "agentsmcp.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "agentsmcp.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the config map
*/}}
{{- define "agentsmcp.configMapName" -}}
{{- printf "%s-config" (include "agentsmcp.fullname" .) }}
{{- end }}

{{/*
Create the name of the secret
*/}}
{{- define "agentsmcp.secretName" -}}
{{- printf "%s-secrets" (include "agentsmcp.fullname" .) }}
{{- end }}

{{/*
Get the image name
*/}}
{{- define "agentsmcp.image" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.image.registry -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end -}}
{{- end }}

{{/*
Get pull secrets
*/}}
{{- define "agentsmcp.imagePullSecrets" -}}
{{- $pullSecrets := list }}
{{- if .Values.global.imagePullSecrets }}
{{- $pullSecrets = .Values.global.imagePullSecrets }}
{{- else if .Values.image.pullSecrets }}
{{- $pullSecrets = .Values.image.pullSecrets }}
{{- end }}
{{- if $pullSecrets }}
imagePullSecrets:
{{- range $pullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Storage class
*/}}
{{- define "agentsmcp.storageClass" -}}
{{- $storageClass := .Values.global.storageClass | default .Values.persistence.storageClass -}}
{{- if $storageClass }}
storageClassName: {{ $storageClass }}
{{- end }}
{{- end }}

{{/*
Namespace
*/}}
{{- define "agentsmcp.namespace" -}}
{{- default .Release.Namespace .Values.namespaceOverride }}
{{- end }}

{{/*
Environment variables from values
*/}}
{{- define "agentsmcp.envVars" -}}
{{- range $key, $value := .Values.env }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}

{{/*
Validate required values
*/}}
{{- define "agentsmcp.validateValues" -}}
{{- if not .Values.image.repository -}}
{{- fail "image.repository is required" -}}
{{- end -}}
{{- if not .Values.image.tag -}}
{{- fail "image.tag is required" -}}
{{- end -}}
{{- end }}

{{/*
Pod security context
*/}}
{{- define "agentsmcp.podSecurityContext" -}}
{{- if .Values.securityContext }}
securityContext:
  {{- toYaml .Values.securityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Container security context
*/}}
{{- define "agentsmcp.containerSecurityContext" -}}
{{- if .Values.podSecurityContext }}
securityContext:
  {{- toYaml .Values.podSecurityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Resources
*/}}
{{- define "agentsmcp.resources" -}}
{{- if .Values.resources }}
resources:
  {{- toYaml .Values.resources | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Node selector
*/}}
{{- define "agentsmcp.nodeSelector" -}}
{{- if .Values.nodeSelector }}
nodeSelector:
  {{- toYaml .Values.nodeSelector | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Tolerations
*/}}
{{- define "agentsmcp.tolerations" -}}
{{- if .Values.tolerations }}
tolerations:
  {{- toYaml .Values.tolerations | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Affinity
*/}}
{{- define "agentsmcp.affinity" -}}
{{- if .Values.affinity }}
affinity:
  {{- toYaml .Values.affinity | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Volume mounts
*/}}
{{- define "agentsmcp.volumeMounts" -}}
- name: tmp
  mountPath: /tmp
{{- if .Values.volumes.data.enabled }}
- name: app-data
  mountPath: /app/data
{{- end }}
{{- if .Values.volumes.logs.enabled }}
- name: app-logs
  mountPath: /app/logs
{{- end }}
{{- if .Values.volumes.temp.enabled }}
- name: app-temp
  mountPath: /app/temp
{{- end }}
{{- if .Values.persistence.enabled }}
- name: persistent-storage
  mountPath: /app/persistent
{{- end }}
{{- end }}

{{/*
Volumes
*/}}
{{- define "agentsmcp.volumes" -}}
- name: tmp
  emptyDir:
    sizeLimit: {{ .Values.volumes.tmp.sizeLimit | default "100Mi" }}
{{- if .Values.volumes.data.enabled }}
- name: app-data
  emptyDir:
    sizeLimit: {{ .Values.volumes.data.sizeLimit | default "500Mi" }}
{{- end }}
{{- if .Values.volumes.logs.enabled }}
- name: app-logs
  emptyDir:
    sizeLimit: {{ .Values.volumes.logs.sizeLimit | default "200Mi" }}
{{- end }}
{{- if .Values.volumes.temp.enabled }}
- name: app-temp
  emptyDir:
    sizeLimit: {{ .Values.volumes.temp.sizeLimit | default "100Mi" }}
{{- end }}
{{- if .Values.persistence.enabled }}
- name: persistent-storage
  persistentVolumeClaim:
    claimName: {{ include "agentsmcp.fullname" . }}-pvc
{{- end }}
{{- end }}