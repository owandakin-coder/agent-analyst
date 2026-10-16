type Lease = { worker_id?: string | null; status?: string; lease_expires_at?: string | null; requested_mode?: string };
export function ownsLease(request: Lease | null, workerId: string | undefined, status = "running", now = Date.now()): boolean {
  if (!request || !workerId || request.worker_id !== workerId || request.status !== status || request.requested_mode !== "paper") return false;
  const expires = Date.parse(request.lease_expires_at ?? "");
  return Number.isFinite(expires) && expires > now;
}
