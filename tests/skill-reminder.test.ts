import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import type { Part } from "@opencode-ai/sdk";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  cosineSimilarity,
  defaultSkillsDirs,
  getSharedExtractor,
  loadSkillRecords,
  parseSkillRecord,
  SkillCache,
} from "../src/skill-cache.ts";
import { createSkillReminderPlugin } from "../src/index.ts";

const TEMP_ROOT = mkdtempSync(join(tmpdir(), "skill-reminder-"));
const SKILLS_DIR = join(TEMP_ROOT, "skills");
const DUPLICATE_SKILLS_DIR = join(TEMP_ROOT, "skills-duplicate");
const INVALID_SKILLS_DIR = join(TEMP_ROOT, "skills-invalid");
const ORIGINAL_SKILLS_DIRS = process.env.REMINDER_INJECTION_SKILLS_DIRS;

function writeSkill(root: string, name: string, description: string, body = `# ${name}\n`) {
  const dir = join(root, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, "SKILL.md"),
    `---\nname: ${name}\ndescription: "${description}"\n---\n\n${body}`,
  );
}

beforeAll(() => {
  writeSkill(
    SKILLS_DIR,
    "justfile",
    "Use when working with just command runner recipes, automation, build orchestration, and task runner maintenance.",
  );
  writeSkill(
    SKILLS_DIR,
    "zotero",
    "Use when managing bibliography entries, DOI metadata, citation cleanup, and reference exports.",
  );
  writeSkill(
    SKILLS_DIR,
    "ast-grep",
    "Use when searching code structure with AST patterns, imports, classes, functions, and structural matches.",
  );
  writeSkill(
    SKILLS_DIR,
    "latex-compile-qa",
    "Use when compiling LaTeX, fixing undefined references, broken citations, bibliography issues, and PDF build failures.",
  );
  writeSkill(
    SKILLS_DIR,
    "ntfy",
    "Use when sending push notifications, ntfy topics, HTTP publish requests, and device alerts.",
  );
  writeSkill(
    SKILLS_DIR,
    "opencode-cli",
    "Use when running OpenCode CLI commands, listing sessions, checking tool availability, inspecting agents, and querying OpenCode runtime state.",
  );
  writeSkill(
    DUPLICATE_SKILLS_DIR,
    "justfile",
    "Duplicate skill that should be ignored when a canonical justfile skill already exists.",
  );
  mkdirSync(join(INVALID_SKILLS_DIR, "broken"), { recursive: true });
  writeFileSync(join(INVALID_SKILLS_DIR, "broken", "SKILL.md"), "# missing frontmatter\n");
});

afterAll(() => {
  if (ORIGINAL_SKILLS_DIRS === undefined) delete process.env.REMINDER_INJECTION_SKILLS_DIRS;
  else process.env.REMINDER_INJECTION_SKILLS_DIRS = ORIGINAL_SKILLS_DIRS;
  rmSync(TEMP_ROOT, { recursive: true, force: true });
});

describe("skill-cache primitives", () => {
  it("parses a skill record from frontmatter", () => {
    const record = parseSkillRecord(
      "/tmp/skill/SKILL.md",
      `---\nname: justfile\ndescription: "Use when editing just recipes."\n---\n\n# justfile\n`,
    );

    expect(record).toEqual({
      name: "justfile",
      description: "Use when editing just recipes.",
      path: "/tmp/skill/SKILL.md",
    });
  });

  it("rejects skill files without both name and description", () => {
    const record = parseSkillRecord(
      "/tmp/skill/SKILL.md",
      `---\nname: justfile\n---\n\n# justfile\n`,
    );

    expect(record).toBeUndefined();
  });

  it("uses the explicit environment override for skill roots", () => {
    process.env.REMINDER_INJECTION_SKILLS_DIRS = "~/custom-one:~/custom-two";
    expect(defaultSkillsDirs()).toEqual([
      `${process.env.HOME}/custom-one`,
      `${process.env.HOME}/custom-two`,
    ]);
  });

  it("falls back only to the standard OpenCode skills root", () => {
    delete process.env.REMINDER_INJECTION_SKILLS_DIRS;
    expect(defaultSkillsDirs()).toEqual([`${process.env.HOME}/.config/opencode/skills`]);
  });

  it("deduplicates skills by name and ignores invalid skill files", async () => {
    const records = await loadSkillRecords([SKILLS_DIR, DUPLICATE_SKILLS_DIR, INVALID_SKILLS_DIR]);
    const names = records.map((record) => record.name).sort();

    expect(names).toEqual(["ast-grep", "justfile", "latex-compile-qa", "ntfy", "opencode-cli", "zotero"]);
  });

  it("computes cosine similarity for aligned, opposite, and mismatched vectors", () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBe(1);
    expect(cosineSimilarity([1, 0], [-1, 0])).toBe(-1);
    expect(cosineSimilarity([1, 0], [0, 1])).toBe(0);
    expect(cosineSimilarity([1, 0], [1, 0, 0])).toBe(-1);
  });
});

describe("semantic ranking usefulness", () => {
  async function extractor(texts: string[]): Promise<number[][]> {
    return (await getSharedExtractor())(texts);
  }

  it("returns useful first choices for representative prompts", async () => {
    const cache = new SkillCache({ skillsDirs: [SKILLS_DIR] }, Promise.resolve(extractor));
    const cases = [
      {
        prompt: "Update the build automation and the just recipes for this project.",
        expectedFirst: "justfile",
      },
      {
        prompt: "Search for import declarations and function definitions with a structural AST matcher.",
        expectedFirst: "ast-grep",
      },
      {
        prompt: "Fix the DOI metadata and clean up the bibliography citations in my reference library.",
        expectedFirst: "zotero",
      },
      {
        prompt: "The LaTeX paper does not compile because references and citations are broken.",
        expectedFirst: "latex-compile-qa",
      },
      {
        prompt: "Send a push notification to an ntfy topic when the job completes.",
        expectedFirst: "ntfy",
      },
      {
        prompt: "List all tool names available in this OpenCode session and reply with one tool name per line.",
        expectedFirst: "opencode-cli",
      },
    ] as const;

    for (const example of cases) {
      const top = await cache.topSkillsForPrompt(example.prompt, 3);
      expect(top[0]?.name, example.prompt).toBe(example.expectedFirst);
      expect(top.map((skill) => skill.name).length, example.prompt).toBe(3);
    }
  });

  it("keeps the build-automation prompt focused on build-related skills", async () => {
    const cache = new SkillCache({ skillsDirs: [SKILLS_DIR] }, Promise.resolve(extractor));
    const top = await cache.topSkillsForPrompt(
      "Refactor the justfile, simplify recipe targets, and improve task-runner automation.",
      3,
    );
    const names = top.map((skill) => skill.name);

    expect(names[0]).toBe("justfile");
    expect(names).not.toContain("ntfy");
  });

  it("surfaces opencode-cli for transcript-derived missed skill prompts", async () => {
    const cache = new SkillCache({ skillsDirs: [SKILLS_DIR] }, Promise.resolve(extractor));
    const transcriptPrompts = [
      'List ALL tool names you can use. Reply with one tool name per line, nothing else.',
      'List ALL parameter names of the task tool. Reply with EXACTLY one parameter name per line, nothing else.',
      'If you can see a tool named websearch, reply with ONLY YES. Otherwise reply with ONLY NO.',
    ];

    for (const prompt of transcriptPrompts) {
      const top = await cache.topSkillsForPrompt(prompt, 3);
      const names = top.map((skill) => skill.name);
      expect(names, prompt).toContain("opencode-cli");
    }
  });
});

describe("plugin injection", () => {
  async function extractor(texts: string[]): Promise<number[][]> {
    return (await getSharedExtractor())(texts);
  }

  it("injects a synthetic reminder part with the top semantic matches", async () => {
    const cache = new SkillCache({ skillsDirs: [SKILLS_DIR] }, Promise.resolve(extractor));
    const plugin = await createSkillReminderPlugin(cache)({
      client: {} as never,
      project: {} as never,
      directory: process.cwd(),
      worktree: process.cwd(),
      serverUrl: new URL("http://localhost"),
      $: {} as never,
    });

    const parts: Part[] = [
      {
        id: "part_user",
        sessionID: "ses_test",
        messageID: "msg_test",
        type: "text",
        text: "Update the build automation and the just recipes for this project.",
      },
    ];

    await plugin["chat.message"]!(
      {
        sessionID: "ses_test",
      },
      {
        message: {
          id: "msg_test",
          sessionID: "ses_test",
          role: "user",
          time: { created: Date.now() },
          agent: "opencode-plugin-reminder-injection-proof",
          model: { providerID: "test", modelID: "test" },
        },
        parts,
      },
    );

    const reminder = parts.find(
      (part) => part.type === "text" && part.synthetic && part.metadata?.source === "opencode-plugin-reminder-injection",
    );
    const text = reminder && reminder.type === "text" ? reminder.text : "";

    expect(text).toContain("consider using");
    expect(text).toContain("justfile");
    expect(text).toContain("latex-compile-qa");
  });
});
