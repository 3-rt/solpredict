import { proxyApiRequest } from "../apiProxy";

export const dynamic = "force-dynamic";

export function POST(request: Request) {
  return proxyApiRequest({ path: "/predict", request, method: "POST" });
}
