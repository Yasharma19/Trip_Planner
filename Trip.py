import streamlit as st
from crewai import Agent, Task, Crew, LLM

# from tools.browser_tools import BrowserTools
# from tools.calculator_tools import CalculatorTools
# from tools.search_tools import SearchTools

from textwrap import dedent
from datetime import date

st.markdown("""
<style>
.card {
  background: #1B1F2A;
  border-radius: 16px;
  padding: 18px 20px;
  margin: 8px 0 16px 0;
  box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}
.stDownloadButton > button {
  border-radius: 12px;
  padding: 10px 14px;
  background: linear-gradient(135deg, #6C63FF 0%, #7C59FF 100%);
  color: white;
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Trip Planner — CrewAI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_page_config(page_title="Trip Planner", layout="centered")
st.title("🧭 CrewAI Trip Planner")
st.markdown(
    "Plan **the best trip** with AI agents that research, guide, and craft your itinerary—complete with weather, budget, and local gems.")
with st.sidebar:
    st.markdown("### ✈️ Trip Inputs")
    origin = st.text_input("Origin (city/airport)", help="Where are you traveling from?")
    cities = st.text_input("Cities")
    range = st.text_input("Trip Window")
    interests = st.multiselect(
        "Interests",
        options=["Beaches", "Food", "Museums", "Hiking", "Nightlife", "Photography", "Shopping", "Architecture",
                 "History", "Nature", "Temple"]
    )

    temperature = st.slider("Select Temperature (creativity level)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    st.markdown("---")
    generate = st.button("🚀 Generate Itinerary", type="primary", use_container_width=True)
    # clear_history = st.button("🧹 Clear History", use_container_width=True)
    # if clear_history:
    #     st.session_state.history = []
    #     st.toast("History cleared!", icon="✅")


# Button to trigger generation
# if st.button("Generate Blog"):
if generate:
    llm = LLM(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=temperature
    )

    city_selection_agent = Agent(
        role='City Selection Expert',
        goal='Select the best city based on weather, season, and prices',
        backstory=
        'An expert in analyzing travel data to pick ideal destinations',
        # tools=[
        #     SearchTools.search_internet,
        #     BrowserTools.scrape_and_summarize_website,
        # ],
        verbose=True,
        llm=llm)

    local_expert_agent = Agent(
        role='Local Expert at this city',
        goal='Provide the BEST insights about the selected city',
        backstory="""A knowledgeable local guide with extensive information
            about the city, it's attractions and customs""",
        # tools=[
        #     SearchTools.search_internet,
        #     BrowserTools.scrape_and_summarize_website,
        # ],
        verbose=True,
        llm=llm)

    travel_plan = Agent(

        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and
            packing suggestions for the city""",
        backstory="""Specialist in travel planning and logistics with
            decades of experience""",
        # tools=[
        #     SearchTools.search_internet,
        #     BrowserTools.scrape_and_summarize_website,
        #     CalculatorTools.calculate,
        # ],
        verbose=True,
        llm=llm)

    identify_task = Task(
        description=dedent(f"""
                    Analyze and select the best city only from {cities} for the trip based
                    on specific criteria such as weather patterns, seasonal
                    events, and travel costs. This task involves comparing
                    multiple cities, considering factors like current weather
                    conditions, upcoming cultural or seasonal events, and
                    overall travel expenses.

                    Your final answer must be a detailed
                    report on the chosen city, and everything you found out
                    about it, including the actual flight costs, train costs, weather
                    forecast and attractions.

                    Traveling from: {origin}
                    City Options: {cities}
                    Trip Date: {range}
                    Traveler Interests: {interests}
                """),
        agent=city_selection_agent,
        expected_output="Detailed report on the chosen city including flight costs, weather forecast, and attractions"
    )

    gather_task = Task(
        description=dedent(f"""
                    As a local expert on this city you must compile an
                    in-depth guide for someone traveling there and wanting
                    to have THE BEST trip ever!
                    Gather information about key attractions, local customs,
                    special events, and daily activity recommendations.
                    Find the best spots to go to, the kind of place only a
                    local would know.
                    This guide should provide a thorough overview of what
                    the city has to offer, including hidden gems, cultural
                    hotspots, must-visit landmarks, weather forecasts, and
                    high level costs.

                    The final answer must be a comprehensive city guide,
                    rich in cultural insights and practical tips,
                    tailored to enhance the travel experience.

                    Trip Date: {range}
                    Traveling from: {origin}
                    Traveler Interests: {interests}
                """),
        agent=local_expert_agent,
        context=[identify_task],
        expected_output="Comprehensive city guide including hidden gems, cultural hotspots, and practical travel tips"
    )

    plan_task = Task(
        description=dedent(f"""
                    Expand this guide into a full travel
                    itinerary with detailed per-day plans, including
                    weather forecasts, places to eat, packing suggestions,
                    and a budget breakdown.

                    You MUST suggest actual places to visit, actual hotels
                    to stay and actual restaurants to go to.

                    This itinerary should cover all aspects of the trip,
                    from arrival to departure, integrating the city guide
                    information with practical travel logistics.

                    Your final answer MUST be a complete expanded travel plan,
                    formatted as markdown, encompassing a daily schedule,
                    anticipated weather conditions, recommended clothing and
                    items to pack, and a detailed budget, ensuring THE BEST
                    TRIP EVER. Be specific and give it a reason why you picked
                    each place, what makes them special!

                    Trip Date: {range}
                    Traveling from: {origin}
                    Traveler Interests: {interests}
                """),
        agent=travel_plan,
        context=[gather_task],
        expected_output="Complete expanded travel plan with daily schedule, weather conditions, packing suggestions, and budget breakdown"
    )

    crew = Crew(
        agents=[city_selection_agent, local_expert_agent, travel_plan],
        tasks=[identify_task, gather_task, plan_task],
        verbose=True
    )

    result = crew.kickoff(inputs={"Origin": origin, "City": cities, "Range": range, "Interests": interests})
    print(result)
    with st.spinner("🧠 Agents collaborating on your itinerary..."):
        crew = Crew(agents=[city_selection_agent, local_expert_agent, travel_plan],
                tasks=[identify_task, gather_task, plan_task], verbose=True)
        result = crew.kickoff(
            inputs={"Origin": origin, "City": cities, "Range": range, "Interests": interests})
        st.toast("Itinerary generated!", icon="🎉")

    # Save to history
    # st.session_state.history.append({
    #     "Chosen City — Research Report": topic,
    #     "facts": task1.output,
    #     "summary": task2.output
    # })

    # Display Results
    st.subheader("🧭 Chosen City — Research Report")
    st.markdown(identify_task.output)            # Research Work

    st.subheader("✍️ Local Expert - Deep Guide")
    st.markdown(gather_task.output)            # Blog Post

    st.subheader("✈️ Full Itinerary")
    st.markdown(plan_task.output)  # Blog Post

    # Download button
    blog_text = (f"Origin: {origin}\nCities: {cities}\nRange: {range}\nInterests: {interests}\n\nChosen City"
                 f"- Research Report:\n{identify_task.output}\n\nLocal Expert - Deep Guide:\n{gather_task.output}\n\nFull Itinerary{plan_task.output}")
    st.download_button("📥 Download Full Itinerary", blog_text)

    # st.success("Blog generation complete!")

# streamlit run C:\Users\ysharma\PycharmProjects\PythonProject\TripPlanner\Trip.py
