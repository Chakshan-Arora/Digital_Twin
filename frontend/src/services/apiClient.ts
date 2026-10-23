import type {
  CaseDetail,
  CaseSummary,
  ClearanceRequest,
  ClearanceResponse,
  HealthResponse,
  UncertaintyRequest,
  UncertaintyResponse,
} from '../types/api';

export class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

export function getBaseUrl(): string {
  const url = import.meta.env.VITE_API_BASE_URL;
  if (url && typeof url === 'string') {
    return url.replace(/\/+$/, '');
  }
  return 'http://127.0.0.1:8000';
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const baseUrl = getBaseUrl();
  const url = `${baseUrl}${endpoint}`;

  let res: Response;
  try {
    res = await fetch(url, {
      ...options,
      headers: {
        'Accept': 'application/json',
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers,
      },
    });
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : 'Network error';
    throw new ApiError(`Unable to connect to backend at ${baseUrl}: ${errorMsg}`, 0);
  }

  if (!res.ok) {
    let errorDetail = `Request failed with status ${res.status}`;
    let bodyData: unknown = null;
    try {
      bodyData = await res.json();
      if (bodyData && typeof bodyData === 'object' && 'detail' in bodyData) {
        const detail = (bodyData as { detail: unknown }).detail;
        if (typeof detail === 'string') {
          errorDetail = detail;
        } else if (Array.isArray(detail)) {
          errorDetail = detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
        } else {
          errorDetail = JSON.stringify(detail);
        }
      }
    } catch {
      // not JSON
      try {
        const text = await res.text();
        if (text) errorDetail = text;
      } catch {
        // ignore
      }
    }
    throw new ApiError(errorDetail, res.status, bodyData);
  }

  return (await res.json()) as T;
}

export const apiClient = {
  async checkHealth(): Promise<HealthResponse> {
    return request<HealthResponse>('/health');
  },

  async listCases(): Promise<CaseSummary[]> {
    return request<CaseSummary[]>('/cases');
  },

  async getCaseDetail(caseId: string): Promise<CaseDetail> {
    return request<CaseDetail>(`/cases/${encodeURIComponent(caseId)}`);
  },

  async getMeshObj(caseId: string, structureName: string): Promise<string> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/cases/${encodeURIComponent(caseId)}/structures/${encodeURIComponent(structureName)}/mesh.obj`;
    let res: Response;
    try {
      res = await fetch(url);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Network error';
      throw new ApiError(`Failed to fetch mesh: ${errorMsg}`, 0);
    }
    if (!res.ok) {
      throw new ApiError(`Failed to load mesh for '${structureName}' (Status ${res.status})`, res.status);
    }
    return res.text();
  },

  async computeClearance(caseId: string, req: ClearanceRequest): Promise<ClearanceResponse> {
    return request<ClearanceResponse>(`/cases/${encodeURIComponent(caseId)}/clearance`, {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  async computeUncertainty(caseId: string, req: UncertaintyRequest): Promise<UncertaintyResponse> {
    return request<UncertaintyResponse>(`/cases/${encodeURIComponent(caseId)}/uncertainty`, {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  getVolumeDownloadUrl(caseId: string): string {
    return `${getBaseUrl()}/cases/${encodeURIComponent(caseId)}/volume.nii.gz`;
  },

  getMaskDownloadUrl(caseId: string, structureName: string): string {
    return `${getBaseUrl()}/cases/${encodeURIComponent(caseId)}/structures/${encodeURIComponent(structureName)}/mask.nii.gz`;
  },
};
