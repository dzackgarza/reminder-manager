import { pipeline } from "@huggingface/transformers";

export type SkillRecord = {
  name: string;
  description: string;
  path: string;
  embedding?: number[];
};

type Extractor = (texts: string[]) => Promise<number[][]>;

type SkillCacheOptions = {
  skillsDirs: string[];
  model: string;
};

const DEFAULT_MODEL = process.env.REMINDER_INJECTION_MODEL?.trim() || "mixedbread-ai/mxbai-embed-xsmall-v1";

let sharedExtractor: Promise<Extractor> | undefined;

function expandHome(path: string): string {
  if (!path.startsWith("~/")) return path;
  return `${process.env.HOME ?? ""}/${path.slice(2)}`;
}

function unique<T>(items: T[]): T[] {
  return [...new Set(items)];
}

export function defaultSkillsDirs(): string[] {
  const envDirs = (process.env.REMINDER_INJECTION_SKILLS_DIRS ?? "")
    .split(":")
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (envDirs.length > 0) {
    return unique(envDirs.map(expandHome));
  }
  return unique([expandHome("~/.config/opencode/skills")]);
}

function parseFrontmatterValue(text: string, key: string): string | undefined {
  const match = text.match(new RegExp(`^${key}:\\s*["']?(.+?)["']?$`, "m"));
  return match?.[1]?.trim();
}

export function parseSkillRecord(path: string, text: string): SkillRecord | undefined {
  const name = parseFrontmatterValue(text, "name");
  const description = parseFrontmatterValue(text, "description");
  if (!name || !description) return undefined;
  return { name, description, path };
}

export async function loadSkillRecords(skillsDirs: string[]): Promise<SkillRecord[]> {
  const files = await Promise.all(
    skillsDirs.map(async (dir) => {
      const proc = Bun.spawn(["find", dir, "-name", "SKILL.md", "-type", "f"], {
        stdout: "pipe",
        stderr: "ignore",
      });
      const stdout = await new Response(proc.stdout).text();
      await proc.exited;
      return stdout
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
    }),
  );

  const records = await Promise.all(
    unique(files.flat()).map(async (path) => {
      const file = Bun.file(path);
      if (!(await file.exists())) return undefined;
      return parseSkillRecord(path, await file.text());
    }),
  );

  const byName = new Map<string, SkillRecord>();
  for (const record of records) {
    if (!record) continue;
    if (!byName.has(record.name)) byName.set(record.name, record);
  }
  return [...byName.values()];
}

export function cosineSimilarity(left: number[], right: number[]): number {
  if (left.length !== right.length || left.length === 0) return -1;
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let i = 0; i < left.length; i += 1) {
    dot += left[i] * right[i];
    leftNorm += left[i] * left[i];
    rightNorm += right[i] * right[i];
  }
  if (leftNorm === 0 || rightNorm === 0) return -1;
  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/g)
    .map((token) => token.trim())
    .filter((token) => token.length >= 3);
}

function lexicalOverlapScore(prompt: string, skill: SkillRecord): number {
  const promptTokens = new Set(tokenize(prompt));
  if (promptTokens.size === 0) return 0;
  const skillTokens = new Set(tokenize(`${skill.name} ${skill.description}`));
  if (skillTokens.size === 0) return 0;
  let overlap = 0;
  for (const token of promptTokens) {
    if (skillTokens.has(token)) overlap += 1;
  }
  return overlap / promptTokens.size;
}

async function buildExtractor(model: string): Promise<Extractor> {
  const extractor = await pipeline("feature-extraction", model, {
    dtype: "fp32",
  });
  return async (texts: string[]) => {
    const output = await extractor(texts, {
      pooling: "mean",
      normalize: true,
    });
    return output.tolist() as number[][];
  };
}

export async function getSharedExtractor(model = DEFAULT_MODEL): Promise<Extractor> {
  sharedExtractor ??= buildExtractor(model);
  return sharedExtractor;
}

export class SkillCache {
  private readonly skillsDirs: string[];
  private readonly model: string;
  private readonly extractorPromise: Promise<Extractor>;
  private recordsPromise: Promise<SkillRecord[]> | undefined;

  constructor(options: Partial<SkillCacheOptions> = {}, extractorPromise?: Promise<Extractor>) {
    this.skillsDirs = options.skillsDirs ?? defaultSkillsDirs();
    this.model = options.model ?? DEFAULT_MODEL;
    this.extractorPromise = extractorPromise ?? getSharedExtractor(this.model);
  }

  async listSkills(): Promise<SkillRecord[]> {
    this.recordsPromise ??= this.load();
    return this.recordsPromise;
  }

  async topSkillsForPrompt(prompt: string, topK: number): Promise<SkillRecord[]> {
    const skills = await this.listSkills();
    if (skills.length === 0) return [];
    const extractor = await this.extractorPromise;
    const [promptEmbedding] = await extractor([prompt]);
    return skills
      .map((skill) => ({
        skill,
        score:
          cosineSimilarity(promptEmbedding, skill.embedding ?? []) +
          lexicalOverlapScore(prompt, skill),
      }))
      .sort((left, right) => right.score - left.score)
      .slice(0, topK)
      .map((entry) => entry.skill);
  }

  private async load(): Promise<SkillRecord[]> {
    const skills = await loadSkillRecords(this.skillsDirs);
    if (skills.length === 0) return [];
    const extractor = await this.extractorPromise;
    const embeddings = await extractor(skills.map((skill) => `${skill.name}: ${skill.description}`));
    return skills.map((skill, index) => ({
      ...skill,
      embedding: embeddings[index],
    }));
  }
}
