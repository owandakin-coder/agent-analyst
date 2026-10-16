type Stored = { api_key_encrypted: string | null; secret_key_encrypted: string | null };
export async function readStoredBrokerSecrets(row: Stored | null, decrypt: (value: string | null) => Promise<string>) {
  if (!row) return { apiKey: "", secretKey: "", readable: true };
  try {
    const [apiKey, secretKey] = await Promise.all([decrypt(row.api_key_encrypted), decrypt(row.secret_key_encrypted)]);
    return { apiKey, secretKey, readable: true };
  } catch {
    return { apiKey: "", secretKey: "", readable: false };
  }
}
export function replacementBrokerSecrets(stored: { apiKey: string; secretKey: string; readable: boolean }, newKey: unknown, newSecret: unknown) {
  const apiKey = typeof newKey === "string" ? newKey.trim() : "";
  const secretKey = typeof newSecret === "string" ? newSecret.trim() : "";
  if (!stored.readable && (!apiKey || !secretKey)) throw new Error("Replace both Paper broker keys, then save and verify the connection.");
  return { apiKey: apiKey || stored.apiKey, secretKey: secretKey || stored.secretKey };
}
