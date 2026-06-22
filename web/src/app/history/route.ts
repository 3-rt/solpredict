import { proxyApiRequest } from "../apiProxy";

export const dynamic = "force-dynamic";

export function GET(request: Request) {
  return proxyApiRequest({ path: "/history", request, method: "GET" });
}
